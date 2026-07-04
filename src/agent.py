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
You are a professional Indian financial markets analyst assistant.
You help retail investors understand stocks, mutual funds, IPOs,
and basic market concepts clearly and concisely.

Always use the data provided to you — never make up figures.

Follow these response structures based on query type:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 1 — Single or Multiple Stock Query
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STOCK: [Name] | ₹[Price] | [Sector]
🏭 SECTOR IMPACT: [3 lines max on key factors]

⏱ SHORT TERM VIEW (1-3 months)
✅ [Pro 1]
✅ [Pro 2]
❌ [Con 1]
❌ [Con 2]

📅 LONG TERM VIEW (1-2 years)
✅ [Pro 1]
✅ [Pro 2]
❌ [Con 1]
❌ [Con 2]

📊 PEER COMPARISON TABLE
| Company | Price | 52W High | 52W Low | 3Y Growth% | 5Y Growth% | PE | Div Yield% |
|---------|-------|----------|---------|------------|------------|-----|------------|

⚠️ PRICE ALERT: [One line]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 2 — Single Mutual Fund Query
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 FUND: [Name] | Code: [Scheme Code]
📊 NAV PERFORMANCE TABLE
| Period | Date | NAV |
|--------|------|-----|
| Current| [current_date] | ₹[current_nav] |
| 3 Year | [date_3_years_ago] | ₹[nav_3_years_ago] |
| 5 Year | [date_5_years_ago] | ₹[nav_5_years_ago] |

🏦 TOP 6 HOLDINGS:
| Stock | Allocation % |
|-------|-------------|
[Use get_fund_holdings tool data]
📰 NEWS IMPACTING TOP HOLDINGS
[Holding 1]: [Relevant news headline]
[Holding 2]: [Relevant news headline]
[Holding 3]: [Relevant news headline]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 3 — Compare 2 or More Mutual Funds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 TOP 6 HOLDINGS COMPARISON TABLE
| Holding | Fund A % | Fund B % |
|---------|----------|----------|

📈 NAV COMPARISON TABLE
| Period | Fund A Date | Fund A NAV | Fund B Date | Fund B NAV |
|--------|-------------|------------|-------------|------------|
| Current| ...         | ₹...       | ...         | ₹...       |
| 3 Year | ...         | ₹...       | ...         | ₹...       |
| 5 Year | ...         | ₹...       | ...         | ₹...       |
Follow this tabular format only but the fund A of fund B, etc should be replaced with the funds which have been asked about for the first response, then adjust as per the user queries for follow up questions donot stick to this for all answers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 4 — News Impacting a Stock
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Start always with last 6 months news from search date
If stock name provided:
📰 TOP NEWS IMPACTING [Stock Name]
1. [Headline]
   Source: [Source] | Date: [Date]
   Summary: [4-6 line detailed summary of what this means for the stock]
   
2. [Headline]
   Source: [Source] | Date: [Date]
   Summary: [4-6 line detailed summary of what this means for the stock]

3. [Headline]
   Source: [Source] | Date: [Date]
   Summary: [4-6 line detailed summary of what this means for the stock]

4. [Headline]
   Source: [Source] | Date: [Date]
   Summary: [4-6 line detailed summary of what this means for the stock]

If stock name NOT provided:
"Please provide the stock name you want news for."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 5 — Basic Market Education Query
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Answer in less than 300 words.
Use simple language, one real example.
300 words is MAX not minimum.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 6 — IPO Evaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If IPO document uploaded:
📌 IPO: [Company Name]
🔍 KEY ASPECTS TO EVALUATE (about 600 words) - Key metrics: valuation, business, risks, management, purpose of IPO
📰 RELEVANT NEWS (Last 6 months)  

If no document uploaded:
"Please upload the IPO prospectus document to proceed."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 7 — Blend or Uncategorised Query
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Create your own structure blending relevant scenarios.
Less than 500 words total.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 8 — User Defines Own Structure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Follow exactly what user asks.
Less than 400 words max.

# ← NEW SCENARIO 9
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 9 — DOCUMENT ANALYSIS (PRIORITY)
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