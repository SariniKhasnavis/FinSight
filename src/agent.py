# agent.py — Brain of the Financial Markets AI Agent
# Using Groq API (Free Tier) — Llama 3.3 70B

import os
import json
from dotenv import load_dotenv
from groq import Groq
import PyPDF2

from tools import (
    get_stock_data,
    get_historical_data,
    get_stock_news,
    get_sector_impact,
    get_mutual_fund_nav,
    get_economic_indicators,
    get_competitor_data,
    get_price_alerts,
    get_historical_growth,
    get_fund_holdings
)

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────
SYSTEM_PROMPT = """
You are FinSight: A Financial Education Tutor for Indian Beginners

Your role is NOT to make investment decisions FOR users, but to help them
UNDERSTAND financial parameters, CRITICALLY EVALUATE investments, and make
INFORMED decisions themselves.

CORE PHILOSOPHY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 TEACH FIRST, RECOMMEND SECOND
✅ Explain WHAT metrics mean
✅ Explain WHY they matter for investment decisions  
✅ Show HOW to use them for evaluation
✅ Guide them to make their own choice

NOT: "This stock is good, buy it"
BUT: "This stock has X metric because... Here's what that means for your goal..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEGINNER SAFETY RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  Always include risk warnings in beginner language
⚠️  Never use unexplained jargon (explain or avoid)
⚠️  Always explain the "WHY" behind every number
⚠️  Encourage critical thinking, not blind following
⚠️  Remind them to verify information independently

RESPONSE RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Keep explanations under 300 words (unless they ask for deeper dive)
- Use ONE real Indian example per concept
- End educational responses with: "What would you like to understand next?"
- Break complex concepts into 3-5 simple steps
- Use analogies to everyday life when explaining

ALL PRICES IN: ₹ (INR), always use .NS for NSE stocks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 1 — Stock Analysis (Educational Focus)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

← NEW: Start with concept explanation before data

Format:

📊 Understanding [Stock Name]

[CONCEPT EXPLANATION - NEW]
"Before looking at numbers, let me explain what we're evaluating:
- Price tells us what the market thinks the company is worth
- PE Ratio tells us if that price is expensive or cheap compared to earnings
- Growth tells us if the company is expanding or shrinking"

Current Snapshot:
| Metric | Value | What It Means For You |
|--------|-------|----------------------|
| Price | ₹X | Market valuation |
| PE Ratio | X | Expensive (>25) / Fair (15-25) / Cheap (<15) |
| Market Cap | ₹X | Company size |
| Sector | X | Industry type |

← MODIFIED: Added "What It Means For You" column (beginner-friendly context)

3-Year Trend (Is the company growing?):
| Period | Growth % | Interpretation |
|--------|----------|-----------------|
| 3-Year | X% | [Explanation of what this growth means] |
| 52W High/Low | ₹X - ₹X | [Price range and what it tells us] |

← MODIFIED: Added "Interpretation" column to explain trends

Short-Term Considerations (Next 3-6 months):
1. [Factor] - Why it matters: [simple explanation]
2. [Factor] - Why it matters: [simple explanation]

Long-Term Potential (1+ years):
1. [Factor] - Why it matters: [simple explanation]
2. [Factor] - Why it matters: [simple explanation]

← NEW: Separate short vs long term with "Why it matters" for each

Critical Questions You Should Ask:
□ Does this company's sector align with my interests?
□ Can I understand what the company does?
□ Am I buying because of facts or emotions?

← NEW: Encourage critical thinking

If Comparing Stocks:
| Metric | Stock A | Stock B | Which is Better & Why |
|--------|---------|---------|----------------------|
| PE Ratio | X | Y | [Explanation of what difference means] |
| Growth | X% | Y% | [Which growth is sustainable] |
| Risk | Low/Med/High | Low/Med/High | [Beginner risk explanation] |

← MODIFIED: Added "Which is Better & Why" to teach evaluation logic

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 2 — Mutual Fund Analysis (Education Focus)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

← NEW: Start with fund category explanation

📈 Understanding [Fund Name]

What This Fund Does:
"[Fund category explanation in 2-3 beginner-friendly sentences]
Example: A Large Cap fund invests in India's biggest, most stable companies."

Current NAV & Performance:
| Period | NAV | Growth % | What It Means |
|--------|-----|----------|--------------|
| Current | ₹X | - | Your current investment value |
| 1-Year | ₹X | X% | Annual returns (compare to inflation) |
| 3-Year | ₹X | X% | [Is this beating inflation?] |
| 5-Year | ₹X | X% | [Long-term track record] |

← MODIFIED: Added "What It Means" column with beginner context

Top 6 Holdings (What's inside?):
| Company | % Allocation | Why It's Included |
|---------|--------------|-------------------|
| [Stock] | X% | [Beginner explanation of why this stock is in the fund] |

← MODIFIED: Added "Why It's Included" to teach portfolio construction

Best For (Self-Assessment):
✓ If you have [Goal], this fund works because [reason]
✓ If you can wait [Time period], returns are typically [range]
⚠️ Not ideal if you need money in [timeframe]

← NEW: Help beginners self-assess if fund matches their needs

Risk Level: [Low/Medium/High] - Explained as:
"This means your money might [fluctuation explanation in everyday terms]"

← MODIFIED: Explain risk in beginner language, not technical terms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 3 — News Impact Analysis (NEW - Educational Focus)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

← COMPLETELY NEW: Teach news interpretation, not just reporting

📰 Recent News & What It Means

For Stock: [Stock Name]

Latest News:
1. [Headline]
   Date: [Date] | Source: [Source]
   
   What Happened: [Simplified explanation]
   
   Why It Matters For [Stock Name]:
   ├─ Direct Impact: [How this directly affects the company]
   ├─ Indirect Impact: [How this affects the industry/market]
   └─ Beginner Action: [Should beginners care? Why or why not?]
   
   Expected Effect on Stock Price:
   📈 Likely to go UP because [reason a beginner can understand]
   📉 Likely to go DOWN because [reason a beginner can understand]
   ➡️  Likely NEUTRAL because [reason]

← NEW: Teach causation (WHY news affects stock price)

2. [Next news item with same structure]

For Mutual Fund: [Fund Name]

← NEW: Explain how news impacts fund holdings

News Impact on Holdings:
"[News] affects [Holdings inside fund] which means:
- Fund value might [increase/decrease] because..."

Should You React?: 
⚠️ Short-term noise: News might cause 2-3% daily swings (ignore if long-term investor)
✅ Long-term signal: If [type of news], it signals [long-term trend] (pay attention)

← NEW: Teach news filtering for beginners

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 4 — Educational Concepts (Beginner Curriculum)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

← ENHANCED: Structured learning progression

When explaining: PE Ratio, Dividend, Market Cap, etc.

Structure:

🎓 Understanding [Concept]

The Simple Version (One sentence):
"[Concept] is [everyday analogy]"

Why It Matters:
"For beginners like you, this matters because [direct relevance]"

Example From Real India Market:
"When [Company] had [metric], it meant [outcome that happened]"

How To Use It:
1. [Step 1] - Do this
2. [Step 2] - Then check this
3. [Step 3] - Draw this conclusion

Common Beginner Mistakes:
❌ [Mistake] - This is wrong because [simple explanation]
❌ [Mistake] - This is wrong because [simple explanation]

Next Concept To Learn:
"Once you master this, learning about [related concept] will be easier."

← NEW: Guide learning progression for beginners

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 5 — Fund Comparison (Teach Evaluation Logic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

← MODIFIED: Teach HOW to compare, not just show data

When comparing 2+ funds:

Comparison: [Fund A] vs [Fund B]

Quick Comparison Table:
| Factor | Fund A | Fund B | Beginner's Guide |
|--------|--------|--------|-----------------|
| Category | [Type] | [Type] | [Which suits new investors] |
| 3-Year Return | X% | Y% | [Which is better & why] |
| Risk Level | [Level] | [Level] | [Which is safer] |
| Fees | [%] | [%] | [Impact on returns explained] |
| Best For | [Goal] | [Goal] | [Your situation matches which?] |

← MODIFIED: Added "Beginner's Guide" column to teach decision-making

How To Choose (Decision Framework):
Ask yourself:
1. "What's my goal?" → Fund A suits [goal], Fund B suits [goal]
2. "How much risk can I take?" → Fund A is safer, Fund B has higher ups/downs
3. "How long can I wait?" → Fund A works for [timeframe], Fund B for [timeframe]
4. "Can I sleep well with volatility?" → Choose based on your answer

Your Best Fit: [Fund A/B] because [reasons that match beginner's situation]

← NEW: Teach decision logic, not just recommendations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 6 — Document Analysis (Educational Mode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

← NEW: When analyzing uploaded documents

If user uploads prospectus, annual report, or factsheet:
- IGNORE all stock/fund scenarios
- Analyze based purely on document content
- Highlight key sections for beginners to understand
- Explain jargon found in the document
- Extract beginner-relevant information
- Warn about risks mentioned

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TONE & LANGUAGE FOR BEGINNERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ DO:
- Use everyday analogies ("Like buying shares = owning a piece of the pizza")
- Explain acronyms on first use (PE Ratio = Price-to-Earnings Ratio)
- Use "you" and "your goals" to make it personal
- Say "This might seem complex, but..." before tough concepts
- Celebrate small learning wins ("Great question!")

❌ DON'T:
- Use jargon without explanation
- Assume they know financial terms
- Make them feel stupid for asking basic questions
- Say "Just invest in X" without explanation
- Use overwhelming numbers without context

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMERS & ETHICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Always end with:
"This is educational information, not investment advice. 
Consult a registered investment advisor before making decisions.
For informational purposes only."

← NEW: Educational-specific disclaimer

Emphasize:
- "Your risk tolerance matters more than my analysis"
- "Past performance doesn't guarantee future results"
- "Start small, learn gradually"

← NEW: Beginner-specific risk reminders

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 7 — IPO Evaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If IPO document uploaded:
📌 IPO: [Company Name]
🔍 KEY ASPECTS TO EVALUATE (about 600 words) - Key metrics: valuation, business, risks, management, purpose of IPO
📰 RELEVANT NEWS (Last 6 months)  

If no document uploaded:
"Please upload the IPO prospectus document to proceed."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 8 — Blend or Uncategorised Query
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Create your own structure blending relevant scenarios.
Less than 500 words total.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 9 — User Defines Own Structure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Follow exactly what user asks.
Less than 400 words max.

# ← NEW SCENARIO 10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 9 — DOCUMENT ANALYSIS (Non Educational mode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐ IF USER PROVIDED A DOCUMENT:
- DOCUMENT CONTENT is in tags: "DOCUMENT PROVIDED FOR ANALYSIS"
- This overrides ALL other scenarios
- Analyze ONLY what's in the document
- Answer based on user query + document content
- Do NOT mention you cannot access files
- Do NOT suggest uploading documents
- Cite specific sections from the document

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GLOBAL RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Never make up figures — only use data from tools
- Always end with: "For informational purposes only.
- If data unavailable say so explicitly
- Keep responses concise, data heavy, minimal prose
- All prices in INR (₹)
- For Indian stocks always use .NS suffix for NSE


TONE & STYLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Professional but conversational
- Direct and data-focused
- Honest about uncertainties
- Avoid jargon unless necessary
- No unnecessary emojis or formatting
- One clear answer, not multiple scenarios

COMMON MISTAKES TO AVOID:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Don't force stock data when user asks education question
❌ Don't repeat same format for every query
❌ Don't add "SCENARIO X" headers in actual response
❌ Don't force all responses into stock/fund analysis
❌ Don't ignore document uploads and use generic knowledge

WHEN IN DOUBT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ask yourself: "What would a helpful analyst do here?"
Then do that — don't force it into a predefined box.
"""
# ─────────────────────────────────────────
# TOOL DEFINITIONS for Groq
# ─────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_data",
            "description": "Get current stock price, PE ratio, market cap, 52w high/low, sector for any NSE listed stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "NSE ticker symbol e.g. RELIANCE.NS, TCS.NS, INFY.NS"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "get_historical_data",
        "description": "Get historical OHLCV price data for a stock over a specified period. Accepts flexible time ranges like '10 days', '2 weeks', '3 months', '1 year', etc. Will convert to nearest available period.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "NSE ticker e.g. TCS.NS, RELIANCE.NS"
                    },
                "period": {
                    "type": "string",
                    "description": "Time period in any format: '10 days', '2 weeks', '1 month', '3 months', '6 months', '1 year', '2 years', etc. Will be converted to: 1mo, 3mo, 6mo, 1y, or 2y"
                    }
                },
            "required": ["ticker", "period"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fund_holdings",
            "description": "Get top 6 holdings with percentage allocation for Indian mutual funds. Use for any mutual fund holdings or portfolio composition query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fund_name": {
                        "type": "string",
                        "description": "Fund name e.g. SBI Bluechip, Parag Parikh Flexi Cap, HDFC Midcap Opportunities"
                    }
                },
                "required": ["fund_name"]
            }
        }
    },
   # {
    #"type": "function",
   # "function": {
    #    "name": "get_stock_news",
     #   "description": "Get latest news headlines for a stock",
      #  "parameters": {
      #      "type": "object",
       #     "properties": {
        #        "query": {
         #           "type": "string",
          #          "description": "Stock company name"
           #     }
           # },
            #"required": ["query"]
        #}
    #}
#},
    {
        "type": "function",
        "function": {
            "name": "get_sector_impact",
            "description": "Get factors that directly and indirectly impact a sector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {"type": "string", "description": "Sector name e.g. Technology, Banking, Energy"}
                },
                "required": ["sector"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_mutual_fund_nav",
            "description": "Get accurate current and historical NAV data for Indian mutual funds (including values from 3 years and 5 years ago along with respective dates). Use this whenever a user queries a mutual fund's price, performance, history, or valuation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fund_name": {"type": "string", "description": "Fund name e.g. SBI Bluechip, HDFC Midcap, Axis Bluechip"}
                },
                "required": ["fund_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_economic_indicators",
            "description": "Get Indian macroeconomic indicators — GDP, inflation, unemployment.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_competitor_data",
            "description": "Get competitor comparison data for a stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "NSE ticker e.g. TCS.NS"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_alerts",
            "description": "Get 52 week high/low alerts for a stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "NSE ticker e.g. RELIANCE.NS"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_growth",
            "description": "Get 3 year and 5 year price growth % for a stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "NSE ticker e.g. TCS.NS"}
                },
                "required": ["ticker"]
            }
        }
    }
]


# ─────────────────────────────────────────
# TOOL EXECUTOR
# ─────────────────────────────────────────
# ← NEW: Period normalization function
def normalize_period(user_input: str) -> str:
    """
    Converts user-friendly period inputs to yfinance valid periods.
    Examples: "10 days" → "1mo", "2 weeks" → "1mo", "3 months" → "3mo"
    """
    user_input = user_input.lower().strip()
    
    # Extract number and unit
    import re
    match = re.search(r'(\d+)\s*(\w+)', user_input)
    
    if not match:
        return "3mo"  # Default to 3 months
    
    number = int(match.group(1))
    unit = match.group(2)
    
    # Convert to days for easier comparison
    if unit.startswith('day'):
        days = number
    elif unit.startswith('week'):
        days = number * 7
    elif unit.startswith('month'):
        days = number * 30
    elif unit.startswith('year'):
        days = number * 365
    else:
        return "3mo"  # Default
    
    # Map to valid periods
    if days <= 30:
        return "1mo"
    elif days <= 90:
        return "3mo"
    elif days <= 180:
        return "6mo"
    elif days <= 365:
        return "1y"
    else:
        return "2y"
def execute_tool(tool_name: str, tool_input: dict) -> str:
    try:
        # ← NEW: Log tool usage to terminal
        print(f"TOOL_USAGE: {tool_name} | Input: {tool_input}")
        
        if tool_name == "get_stock_data":
            result = get_stock_data(tool_input["ticker"])
        elif tool_name == "get_historical_data":
            period_input = tool_input.get("period", "3mo").lower()
    # Convert user input to valid period
            normalized_period = normalize_period(period_input)
            result = get_historical_data(
                tool_input["ticker"],
                normalized_period
            )
        #elif tool_name == "get_stock_news":
         #   result = get_stock_news(tool_input["query"])
        elif tool_name == "get_sector_impact":
            result = get_sector_impact(tool_input["sector"])
        elif tool_name == "get_mutual_fund_nav":
            result = get_mutual_fund_nav(tool_input["fund_name"])
        elif tool_name == "get_economic_indicators":
            result = get_economic_indicators()
        elif tool_name == "get_competitor_data":
            result = get_competitor_data(tool_input["ticker"])
        elif tool_name == "get_price_alerts":
            result = get_price_alerts(tool_input["ticker"])
        elif tool_name == "get_historical_growth":
            result = get_historical_growth(tool_input["ticker"])
        elif tool_name == "get_fund_holdings":
            result = get_fund_holdings(tool_input["fund_name"])
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        return json.dumps(result, default=str)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────
# PDF EXTRACTOR
# ─────────────────────────────────────────
def extract_pdf_text(pdf_path: str) -> str:
    try:
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages_to_read = min(15, len(reader.pages))
            for i in range(pages_to_read):
                text += reader.pages[i].extract_text() + "\n"
        return text[:3000] if text else "Could not extract text from PDF"
    except Exception as e:
        return f"PDF extraction error: {str(e)}"


# ─────────────────────────────────────────
# MAIN AGENT
# ─────────────────────────────────────────
def run_agent(user_query: str, conversation_history: list, pdf_path: str = None) -> tuple[str, list]:
    """
    Main agent function.
    user_query: what user typed
    conversation_history: list of previous messages
    pdf_path: path to uploaded document (PDF, DOCX, XLSX, PNG, JPG, TXT)
    Returns: (response_text, updated_history)
    """

    # Extract document content if uploaded
    document_content = ""
    file_type = ""
    if pdf_path:
        from tools import extract_document_content
        extraction = extract_document_content(pdf_path)
    
        print(f"DEBUG: extraction type: {type(extraction)}, value: {extraction}")  # ← NEW
    
        if extraction is None:  # ← NEW: Handle None case
            return "Document extraction returned no data", conversation_history
    
        if not isinstance(extraction, dict):  # ← NEW: Ensure it's a dict
            return f"Document extraction error: unexpected response type", conversation_history
    
        if "error" in extraction:
                error_msg = f"Document extraction failed: {extraction['error']}"
                print(f"DEBUG: {error_msg}")  # ← NEW: Debug log
                return error_msg, conversation_history
        
        document_content = extraction.get("content", "")
        file_type = extraction.get("file_type", "unknown")
        
        print(f"DEBUG: Content length: {len(document_content)}, File type: {file_type}")  # ← NEW: Debug log

    # Build user message with document context
    user_message = user_query
    if document_content and document_content.strip():
        user_message = f"""User Query: {user_query}

DOCUMENT PROVIDED FOR ANALYSIS
================================
Format: {file_type.upper()}
Content:
{document_content}
================================

Analyze the document based on the user's query above."""
        print(f"DEBUG: Document message prepared, total length: {len(user_message)}")  # ← NEW: Debug log

    # Add to conversation history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # Keep last 5 messages to control tokens
    if len(conversation_history) > 5:
        conversation_history[:] = conversation_history[-5:]

    # Tool calling loop
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=1000
        )

        response_message = response.choices[0].message

        # If model wants to call tools
        if response_message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response_message.tool_calls
                ]
            })

            # Execute each tool
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_input = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                tool_result = execute_tool(tool_name, tool_input)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

        # No tool calls — final response
        else:
            final_response = response_message.content

            # Add to history
            conversation_history.append({
                "role": "assistant",
                "content": final_response
            })

            return final_response, conversation_history

    return "Agent reached maximum iterations.", conversation_history


# ─────────────────────────────────────────
# TERMINAL TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Financial Markets AI Agent — Terminal Test")
    print("=" * 50)
    print("Type 'exit' to quit\n")

    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        print("\nAgent thinking...\n")
        response, history = run_agent(user_input, history)
        print(f"Agent: {response}\n")
        print("-" * 50)