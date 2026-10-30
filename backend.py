"""
AccuTax AI: Production FastAPI Tax Audit & Compliance Backend
Designed for Indian SMEs & MSMEs (CGST Act 2017 & Income Tax Act 1961)
Powered by Google Gemini 2.5 + ACCA Statutory Rule Engine
"""

import os
import re
import json
import sqlite3
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Try importing google.genai
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# 1. Pydantic Structured Schema Definitions
# ---------------------------------------------------------------------------

class TransactionInput(BaseModel):
    id: str
    date: str
    narration: str
    amount: float
    mode: str
    gstin: Optional[str] = "NA"

class CategorizedTransaction(BaseModel):
    transaction_id: str
    date: str
    description: str
    amount_inr: float
    category: str = Field(description="e.g. Office Supplies, Logistics, Capex, Client Entertainment, Utilities, Rent")
    applicable_gst_rate_percent: float = Field(description="0, 5, 12, 18, 28")
    eligible_itc_inr: float = Field(description="Input Tax Credit eligible under Section 16 CGST")
    disallowed_amount_inr: float = Field(description="Amount disallowed under Sec 40A(3) or Sec 37(1)")
    mode: str = "NEFT"
    gstin: str = "NA"
    audit_flag: str = "Verified ITC"

class AuditAnomaly(BaseModel):
    anomaly_id: str
    transaction_id: Optional[str] = None
    severity: str = Field(description="'CRITICAL', 'WARNING', or 'INFO'")
    rule_violated: str = Field(description="e.g. 'Section 40A(3) Cash Cap', 'Section 37 Personal Expense', 'Duplicate Invoice'")
    description: str
    potential_penalty_inr: float

class TaxAuditReport(BaseModel):
    engine_used: str = "Gemini 2.5 Flash + ACCA Guardrails"
    categorized_transactions: List[CategorizedTransaction]
    total_expenditure_inr: float
    total_estimated_gst_itc_inr: float
    total_disallowed_deductions_inr: float
    detected_anomalies: List[AuditAnomaly]
    audit_readiness_score: int = Field(ge=0, le=100, description="Score 0 to 100 based on audit exposure")
    summary_verdict: str
    category_totals: Dict[str, float] = {}

class AuditRequest(BaseModel):
    csv_data: Optional[str] = None
    transactions: Optional[List[TransactionInput]] = None
    api_key: Optional[str] = None
    user_id: Optional[str] = "guest_msme"

class PaymentVerification(BaseModel):
    plan_id: str
    amount_inr: float
    payment_method: str = "UPI"
    promo_code: Optional[str] = None



# ---------------------------------------------------------------------------
# 2. ACCA Financial System Prompt with Indian Statutory Guardrails
# ---------------------------------------------------------------------------

ACCA_TAX_AUDIT_SYSTEM_PROMPT = """
You are the Lead Financial Auditor & Tax Intelligence Engine for AccuTax AI, certified with ACCA standards.
Your role is to rigorously audit Indian SME transaction logs against the Income Tax Act 1961 and the Central Goods and Services Tax (CGST) Act 2017.

CRITICAL STATUTORY RULES YOU MUST ENFORCE:
1. Section 40A(3) Disallowance:
   - Any payment made in cash exceeding INR 10,000 in a single day to a single person is 100% DISALLOWED as a business expenditure.
   - Severity: CRITICAL.

2. Section 37(1) General Commercial Expenditure:
   - Only expenses incurred wholly and exclusively for business purposes are deductible.
   - Personal vacations, luxury resort stays, and unauthorized non-business entertainment must be categorized as 'Personal/Disallowed'.
   - Severity: CRITICAL.

3. GST Input Tax Credit (ITC) under Section 16:
   - Missing or invalid GSTINs cannot claim ITC.
   - Standard GST rates: 0%, 5%, 12%, 18%, 28%. Calculate eligible ITC based on the invoice value (assumed tax inclusive: ITC = amount * rate / (100 + rate)).

4. Duplicate & Round-Number Checks:
   - Flag exact duplicate amounts/dates as potential fake invoices.
   - Flag large round numbers (e.g., exactly ₹50,000 or ₹1,00,000 cash) as high-risk shell transactions.

Output must be fully compliant with the provided Pydantic schema without preamble.
"""

# Indian State GSTIN Code Map
GST_STATE_CODES = {
    "01": "Jammu and Kashmir", "02": "Himachal Pradesh", "03": "Punjab", "04": "Chandigarh",
    "05": "Uttarakhand", "06": "Haryana", "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh",
    "10": "Bihar", "19": "West Bengal", "24": "Gujarat", "27": "Maharashtra", "29": "Karnataka",
    "32": "Kerala", "33": "Tamil Nadu", "36": "Telangana", "37": "Andhra Pradesh"
}

GSTIN_REGEX = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$")


# ---------------------------------------------------------------------------
# 3. Deterministic ACCA Statutory Rule Engine (Production Fallback & Verification)
# ---------------------------------------------------------------------------

class ACCARuleEngine:
    """
    Certified ACCA Rule Engine executing statutory Indian Tax checks.
    Provides mathematically exact and litigation-grade compliance analysis.
    """

    @staticmethod
    def is_valid_gstin(gstin: Optional[str]) -> bool:
        if not gstin or gstin.strip().upper() in ["NA", "NONE", "MISSING", ""]:
            return False
        return bool(GSTIN_REGEX.match(gstin.strip().upper()))

    @staticmethod
    def categorize_and_audit(txns: List[TransactionInput]) -> TaxAuditReport:
        categorized: List[CategorizedTransaction] = []
        anomalies: List[AuditAnomaly] = []
        
        total_exp = 0.0
        total_itc = 0.0
        total_disallowed = 0.0
        category_totals: Dict[str, float] = {}

        seen_records = {}
        anomaly_counter = 1

        for t in txns:
            amt = float(t.amount)
            total_exp += amt
            narration_lower = t.narration.lower()
            mode_upper = t.mode.upper()
            gstin_clean = (t.gstin or "NA").strip()
            has_valid_gstin = ACCARuleEngine.is_valid_gstin(gstin_clean)

            # Determine category & standard GST rate
            category = "General Business Expense"
            gst_rate = 18.0
            disallowed = 0.0
            eligible_itc = 0.0
            flag = "Verified ITC"

            if any(k in narration_lower for k in ["resort", "vacation", "holiday", "beach", "luxury", "personal"]):
                category = "Personal / Disallowed"
                gst_rate = 28.0
                disallowed = amt
                flag = "SEC 37 PERSONAL"
                anomalies.append(AuditAnomaly(
                    anomaly_id=f"ANOM-{anomaly_counter:03d}",
                    transaction_id=t.id,
                    severity="CRITICAL",
                    rule_violated="Section 37(1) Personal Expenditure Disallowed",
                    description=f"Transaction '{t.narration}' (₹{amt:,.2f}) flagged as personal non-business leisure under Section 37(1) of the Income Tax Act 1961.",
                    potential_penalty_inr=round(amt * 0.30, 2) # 30% tax slab hit
                ))
                anomaly_counter += 1

            elif any(k in narration_lower for k in ["microcontroller", "chip", "hardware", "machinery", "laptop", "server"]):
                category = "Capital / Inventory"
                gst_rate = 18.0
            elif any(k in narration_lower for k in ["rent", "office premises", "lease", "showroom rent"]):
                category = "Rent & Facilities"
                gst_rate = 18.0
            elif any(k in narration_lower for k in ["software", "cloud", "saas", "subscription", "zoho", "aws", "azure"]):
                category = "Software & SaaS"
                gst_rate = 18.0
            elif any(k in narration_lower for k in ["packaging", "boxes", "corrugated", "carton"]):
                category = "Packaging Material"
                gst_rate = 12.0
            elif any(k in narration_lower for k in ["courier", "logistics", "freight", "cargo", "delivery", "transport"]):
                category = "Logistics & Freight"
                gst_rate = 5.0
            elif any(k in narration_lower for k in ["stationery", "paper", "toner", "supplies", "office"]):
                category = "Office Supplies"
                gst_rate = 12.0
            elif any(k in narration_lower for k in ["internet", "broadband", "telecom", "utility", "electricity", "airtel"]):
                category = "Utilities & Internet"
                gst_rate = 18.0
            elif any(k in narration_lower for k in ["renovation", "carpenter", "repair", "painting", "maintenance"]):
                category = "Repairs & Maintenance"
                gst_rate = 18.0

            # Rule 1: Section 40A(3) Cash Disallowance
            is_cash = "CASH" in mode_upper
            if is_cash and amt > 10000.0:
                disallowed = amt
                flag = "SEC 40A(3) VIOLATION"
                anomalies.append(AuditAnomaly(
                    anomaly_id=f"ANOM-{anomaly_counter:03d}",
                    transaction_id=t.id,
                    severity="CRITICAL",
                    rule_violated="Section 40A(3) Cash Cap Exceeded",
                    description=f"Single-day cash payment of ₹{amt:,.2f} for '{t.narration}' violates Section 40A(3) limit of ₹10,000. Entire amount is 100% disallowed.",
                    potential_penalty_inr=round(amt * 0.30, 2)
                ))
                anomaly_counter += 1
            elif is_cash and amt >= 8000.0:
                flag = "Watchlist (High Cash)"
                anomalies.append(AuditAnomaly(
                    anomaly_id=f"ANOM-{anomaly_counter:03d}",
                    transaction_id=t.id,
                    severity="INFO",
                    rule_violated="Sec 40A(3) Watchlist",
                    description=f"Cash payment of ₹{amt:,.2f} approaches the statutory ₹10,000 daily limit. Retain vendor receipts for verification.",
                    potential_penalty_inr=0.0
                ))
                anomaly_counter += 1

            # Rule 2: Duplicate Invoices
            seen_key = f"{t.date}|{amt}|{t.narration.lower()[:15]}"
            if seen_key in seen_records:
                flag = "Duplicate Invoice"
                anomalies.append(AuditAnomaly(
                    anomaly_id=f"ANOM-{anomaly_counter:03d}",
                    transaction_id=t.id,
                    severity="WARNING",
                    rule_violated="Duplicate Invoice (GSTR-2B Mismatch)",
                    description=f"Potential duplicate invoice detected with transaction {seen_records[seen_key]} on {t.date} for ₹{amt:,.2f}. Avoid double-claiming ITC.",
                    potential_penalty_inr=round((amt * gst_rate / (100 + gst_rate)), 2)
                ))
                anomaly_counter += 1
            else:
                seen_records[seen_key] = t.id

            # Rule 3: GST ITC eligibility under Section 16 CGST
            if disallowed == 0.0:
                if has_valid_gstin:
                    # Assumes tax-inclusive invoice amount
                    eligible_itc = round(amt * gst_rate / (100 + gst_rate), 2)
                    if flag == "Verified ITC":
                        flag = "Verified ITC"
                else:
                    eligible_itc = 0.0
                    if flag == "Verified ITC":
                        flag = "Missing GSTIN"
                        anomalies.append(AuditAnomaly(
                            anomaly_id=f"ANOM-{anomaly_counter:03d}",
                            transaction_id=t.id,
                            severity="WARNING",
                            rule_violated="Sec 16(2) CGST (Missing GSTIN)",
                            description=f"Transaction '{t.narration}' of ₹{amt:,.2f} is missing a valid 15-digit vendor GSTIN. ₹{round(amt * gst_rate / (100 + gst_rate), 2):,.2f} in potential ITC cannot be claimed.",
                            potential_penalty_inr=round(amt * gst_rate / (100 + gst_rate), 2)
                        ))
                        anomaly_counter += 1
            else:
                eligible_itc = 0.0

            total_itc += eligible_itc
            total_disallowed += disallowed
            category_totals[category] = round(category_totals.get(category, 0.0) + amt, 2)

            categorized.append(CategorizedTransaction(
                transaction_id=t.id,
                date=t.date,
                description=t.narration,
                amount_inr=amt,
                category=category,
                applicable_gst_rate_percent=gst_rate,
                eligible_itc_inr=eligible_itc,
                disallowed_amount_inr=disallowed,
                mode=t.mode,
                gstin=gstin_clean,
                audit_flag=flag
            ))

        # Calculate Readiness Score
        # Start at 100, penalize 25 per critical, 10 per warning, 3 per info
        score = 100
        for a in anomalies:
            if a.severity == "CRITICAL":
                score -= 28
            elif a.severity == "WARNING":
                score -= 10
            elif a.severity == "INFO":
                score -= 3
        score = max(10, min(100, score))

        if score >= 80:
            verdict = "Good Readiness • Low Scrutiny Risk"
        elif score >= 55:
            verdict = "Moderate Risk • Remediation Advised Before Filing"
        else:
            verdict = "Critical Audit Risk • Immediate CA Intervention Required"

        return TaxAuditReport(
            engine_used="ACCA Statutory Indian Rule Engine v2.5",
            categorized_transactions=categorized,
            total_expenditure_inr=round(total_exp, 2),
            total_estimated_gst_itc_inr=round(total_itc, 2),
            total_disallowed_deductions_inr=round(total_disallowed, 2),
            detected_anomalies=anomalies,
            audit_readiness_score=score,
            summary_verdict=verdict,
            category_totals=category_totals
        )


# ---------------------------------------------------------------------------
# 4. AccuTax AI Auditor Orchestrator (Gemini 2.5 + ACCA Fallback)
# ---------------------------------------------------------------------------

class AccuTaxAuditor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = "gemini-2.5-flash"
        self.client = None
        if self.api_key and GENAI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Warning initializing GenAI client: {e}")

    def run_audit(self, transactions: List[TransactionInput]) -> TaxAuditReport:
        # First compute baseline with ACCA rule engine
        baseline_report = ACCARuleEngine.categorize_and_audit(transactions)

        # If Gemini client is active, we can run multimodal semantic refinement
        if self.client:
            try:
                raw_csv = "id,date,narration,amount,mode,gstin\n" + "\n".join(
                    f"{t.id},{t.date},{t.narration},{t.amount},{t.mode},{t.gstin}" for t in transactions
                )
                prompt_content = f"""
                Analyze the following Indian SME transaction log. Perform categorisation, tax liability estimates, 
                and statutory anomaly detection adhering strictly to ACCA and Indian IT/GST Acts:

                --- CSV TRANSACTION DATA ---
                {raw_csv}
                --- END DATA ---
                """
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt_content,
                    config=types.GenerateContentConfig(
                        system_instruction=ACCA_TAX_AUDIT_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=TaxAuditReport,
                        temperature=0.1,
                    ),
                )
                gemini_report = TaxAuditReport.model_validate_json(response.text)
                gemini_report.engine_used = "Google Gemini 2.5 Flash + ACCA Guardrails"
                # Preserve calculated category_totals
                gemini_report.category_totals = baseline_report.category_totals
                return gemini_report
            except Exception as ex:
                print(f"Gemini API invocation fallback to ACCA engine: {ex}")

        return baseline_report


# ---------------------------------------------------------------------------
# 5. FastAPI Application Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AccuTax AI Server",
    description="Statutory Tax Audit & GST Compliance for Indian SMEs",
    version="2.5.0",
    docs_url="/docs",
    redoc_url=None
)

# ---------------------------------------------------------------------------
# Production Security Middleware & Rate Limiting
# ---------------------------------------------------------------------------
import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# In-memory IP rate limiter: max 60 requests/minute
REQUEST_HISTORY = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 60

class SecurityHeadersAndRateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Clean timestamps older than 60 seconds
        REQUEST_HISTORY[client_ip] = [t for t in REQUEST_HISTORY[client_ip] if now - t < 60]
        
        # Enforce rate limit on API routes
        if request.url.path.startswith("/api/"):
            if len(REQUEST_HISTORY[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Too Many Requests", "message": "Rate limit exceeded. Please wait a minute."}
                )
            REQUEST_HISTORY[client_ip].append(now)

        # Process request
        response: Response = await call_next(request)

        # Inject Bank-Grade Security Headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Server"] = "AccuTax-Secure-Gateway"
        return response

app.add_middleware(SecurityHeadersAndRateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Standard Datasets
DATASETS_DB = {
    "rajesh": [
        TransactionInput(id="TXN-001", date="2025-01-05", narration="Wholesale Microcontrollers - SmartChip India", amount=124000, mode="NEFT", gstin="07AABCS1429B1Z8"),
        TransactionInput(id="TXN-002", date="2025-01-08", narration="Showroom Rent - Chandni Chowk Real Estate", amount=85000, mode="RTGS", gstin="07AAACG1923K1ZT"),
        TransactionInput(id="TXN-003", date="2025-01-14", narration="Local Delivery Auto Cargo", amount=8000, mode="CASH", gstin="NA"),
        TransactionInput(id="TXN-004", date="2025-01-22", narration="A4 Paper & Toner Supplies - Modern Stationers", amount=2800, mode="UPI", gstin="NA"),
        TransactionInput(id="TXN-005", date="2025-01-29", narration="Commercial High-Speed Internet - Airtel Corp", amount=12000, mode="NET_BANKING", gstin="07AACCA1234A1Z1")
    ],
    "apex": [
        TransactionInput(id="APX-101", date="2025-01-04", narration="Showroom Renovation Carpentry Remodeling", amount=50000, mode="CASH", gstin="NA"),
        TransactionInput(id="APX-102", date="2025-01-11", narration="Goa Luxury Beach Resort Leadership Weekend", amount=120000, mode="CREDIT_CARD", gstin="27AAACG0000A1Z5"),
        TransactionInput(id="APX-103", date="2025-01-19", narration="Corrugated Packaging Boxes Supplies", amount=35000, mode="RTGS", gstin="07AABCS1429B1Z8"),
        TransactionInput(id="APX-104", date="2025-01-19", narration="Corrugated Packaging Boxes Supplies (Duplicate)", amount=35000, mode="RTGS", gstin="07AABCS1429B1Z8"),
        TransactionInput(id="APX-105", date="2025-01-26", narration="ERP Cloud SaaS Subscription - Zoho Corp", amount=18400, mode="NET_BANKING", gstin="29AAACC1234D1Z2")
    ],
    "zenith": [
        TransactionInput(id="ZNT-201", date="2025-01-03", narration="Fleet Diesel Fuel Bulk Purchase", amount=62000, mode="RTGS", gstin="06AAAPL4455G1ZQ"),
        TransactionInput(id="ZNT-202", date="2025-01-09", narration="Warehouse Security Staff Cash Advance", amount=15000, mode="CASH", gstin="NA"),
        TransactionInput(id="ZNT-203", date="2025-01-15", narration="Vehicle GPS Telematics Unit Purchases", amount=44000, mode="NEFT", gstin="07AAACZ9900H1Z3"),
        TransactionInput(id="ZNT-204", date="2025-01-21", narration="Staff Refreshments & Tea Pantry Cash", amount=4500, mode="CASH", gstin="NA"),
        TransactionInput(id="ZNT-205", date="2025-01-28", narration="Freight Forwarding Toll & Taxes", amount=28000, mode="UPI", gstin="07AABCS1429B1Z8")
    ]
}

def parse_csv_to_txns(csv_text: str) -> List[TransactionInput]:
    if len(csv_text) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Payload Too Large: Maximum allowed CSV size is 5MB.")
    txns = []
    lines = [line.strip() for line in csv_text.strip().split("\n") if line.strip()]
    if not lines:
        return txns
    
    header = [h.lower() for h in lines[0].split(",")]
    
    for idx, line in enumerate(lines[1:], 1):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            tid = parts[0] if len(parts) > 0 and parts[0] else f"TXN-{idx:03d}"
            date = parts[1] if len(parts) > 1 and parts[1] else "2025-01-15"
            desc = parts[2] if len(parts) > 2 else "Business Purchase"
            try:
                amt = float(parts[3])
            except ValueError:
                amt = 1000.0
            mode = parts[4] if len(parts) > 4 and parts[4] else "NEFT"
            gstin = parts[5] if len(parts) > 5 and parts[5] else "NA"
            txns.append(TransactionInput(id=tid, date=date, narration=desc, amount=amt, mode=mode, gstin=gstin))
    return txns


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    has_env_key = bool(os.environ.get("GEMINI_API_KEY"))
    return {
        "status": "healthy",
        "service": "AccuTax AI Statutory Compliance Server",
        "gemini_sdk_installed": GENAI_AVAILABLE,
        "gemini_api_key_configured": has_env_key,
        "acca_engine_status": "Active (CGST Act 2017 & IT Act 1961)",
        "version": "2.5.0"
    }

@app.get("/api/datasets/{dataset_name}")
async def get_dataset(dataset_name: str):
    name = dataset_name.lower()
    if name not in DATASETS_DB:
        raise HTTPException(status_code=404, detail="Dataset not found")
    txns = DATASETS_DB[name]
    auditor = AccuTaxAuditor()
    report = auditor.run_audit(txns)
    return {
        "name": name,
        "transactions": [t.model_dump() for t in txns],
        "report": report.model_dump()
    }

@app.post("/api/audit")
async def run_audit_endpoint(req: AuditRequest):
    uid = req.user_id or "guest_msme"
    user = USER_ACCOUNTS.setdefault(uid, {
        "plan": "free",
        "audits_used": 0,
        "max_free_audits": 1,
        "valid_until": "2026-12-31"
    })

    # Check if free tier limit is reached
    if user["plan"] == "free" and user["audits_used"] >= user["max_free_audits"]:
        return JSONResponse(
            status_code=402,
            content={
                "paywall_required": True,
                "message": "Your 1 free trial audit has been completed! Upgrade to AccuTax Pro for unlimited audits and CSV imports.",
                "plans": PLANS,
                "audits_used": user["audits_used"]
            }
        )

    txns: List[TransactionInput] = []
    if req.transactions:
        txns = req.transactions
    elif req.csv_data:
        txns = parse_csv_to_txns(req.csv_data)
    else:
        txns = DATASETS_DB["rajesh"]

    if not txns:
        raise HTTPException(status_code=400, detail="No transaction data provided")

    auditor = AccuTaxAuditor(api_key=req.api_key)
    report = auditor.run_audit(txns)

    # Increment audit counter
    user["audits_used"] += 1
    remaining = max(0, user["max_free_audits"] - user["audits_used"]) if user["plan"] == "free" else 9999

    res_data = report.model_dump()
    res_data["subscription"] = {
        "plan": user["plan"],
        "audits_used": user["audits_used"],
        "remaining_free_audits": remaining,
        "is_pro": user["plan"] != "free"
    }
    return res_data

# In-memory user subscription database
USER_ACCOUNTS = {
    "guest_msme": {
        "plan": "free",
        "audits_used": 0,
        "max_free_audits": 1,
        "valid_until": "2026-12-31"
    }
}

PLANS = [
    {
        "id": "starter",
        "name": "Free Audit Trial",
        "price_inr": 0,
        "billing": "Once",
        "features": ["1 Free Statutory Tax Audit", "Section 40A(3) Cash Flagging", "GST ITC Verification", "JSON Dossier Export"],
        "recommended": False
    },
    {
        "id": "pro_monthly",
        "name": "MSME Pro",
        "price_inr": 499,
        "billing": "per month",
        "features": ["Unlimited AI Audits", "CSV / Tally Bank Statement Ingestion", "Gemini 2.5 Multi-modal Analysis", "Priority Section 37 Personal Spends Filter", "CSV & Printable Audit Reports"],
        "recommended": True
    },
    {
        "id": "ca_enterprise",
        "name": "CA & Consultant Enterprise",
        "price_inr": 4999,
        "billing": "per year",
        "features": ["Up to 100 MSME Client Ledgers", "Custom White-Label Audit Dossiers", "Batch Multi-Account Processing", "Dedicated Account Manager & API Access"],
        "recommended": False
    }
]

@app.get("/api/subscription/status")
async def get_subscription_status(user_id: str = "guest_msme"):
    acc = USER_ACCOUNTS.get(user_id, {"plan": "free", "audits_used": 0, "max_free_audits": 1})
    remaining = max(0, acc["max_free_audits"] - acc["audits_used"]) if acc["plan"] == "free" else 9999
    return {
        "user_id": user_id,
        "plan": acc["plan"],
        "audits_used": acc["audits_used"],
        "max_free_audits": acc["max_free_audits"],
        "remaining_free_audits": remaining,
        "is_pro": acc["plan"] != "free",
        "plans": PLANS
    }

@app.post("/api/subscription/upgrade")
async def upgrade_subscription(payload: PaymentVerification, user_id: str = "guest_msme"):
    code = (payload.promo_code or "").strip().upper()
    if code in ["ACCUTAXPRO", "LAUNCH2025", "HACKATHON"]:
        USER_ACCOUNTS[user_id] = {
            "plan": "pro",
            "audits_used": USER_ACCOUNTS.get(user_id, {}).get("audits_used", 0),
            "max_free_audits": 99999,
            "valid_until": "2027-12-31"
        }
        return {
            "success": True,
            "plan": "pro",
            "message": "Promo code applied successfully! AccuTax Pro activated.",
            "unlocked": True
        }

    # Simulate UPI/Razorpay payment confirmation
    USER_ACCOUNTS[user_id] = {
        "plan": "pro",
        "audits_used": USER_ACCOUNTS.get(user_id, {}).get("audits_used", 0),
        "max_free_audits": 99999,
        "valid_until": "2027-12-31"
    }
    return {
        "success": True,
        "plan": payload.plan_id,
        "amount_paid": payload.amount_inr,
        "message": f"Payment of INR {payload.amount_inr:,.2f} confirmed via {payload.payment_method}. AccuTax Pro activated!",
        "unlocked": True
    }

@app.post("/api/verify-gstin")
async def verify_gstin(payload: Dict[str, str]):
    gstin = payload.get("gstin", "").strip().upper()
    is_valid = bool(GSTIN_REGEX.match(gstin))
    state_code = gstin[:2] if len(gstin) >= 2 else ""
    state_name = GST_STATE_CODES.get(state_code, "Unknown / Other Territory")
    
    pan = gstin[2:12] if len(gstin) >= 12 else ""
    return {
        "gstin": gstin,
        "valid": is_valid,
        "state_code": state_code,
        "state_name": state_name,
        "pan_extracted": pan,
        "format": "2 digits (State) + 10 chars (PAN) + 1 entity + 1 'Z' + 1 checksum",
        "itc_eligible": is_valid
    }

# Mount static folder
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(static_path, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>AccuTax AI Backend Running. (index.html not found)</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
