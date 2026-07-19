# agent.py — Brain of the Financial Markets AI Agent
# Using Groq API (Free Tier) — Llama 3.3 70B

import os
import json
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
from src.tools import extract_document_content

from src.tools import (
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

🎓 CORE PHILOSOPHY:
- Teach concepts clearly in beginner language
- Add "What It Means For You" context
- Use real tool data, never placeholders
- Follow scenario structure consistently

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 SCENARIO 1: Stock Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IF user provides chart data (RSI, MACD, Price, etc):
- Analyze using the provided metrics
- Explain what each metric means for beginners
- NO need to call tools (data already provided)

IF user asks without chart data:
- Call get_stock_data() for fundamentals
- Call get_historical_growth() for 3-year growth
- Create table with real data
- Explain each metric in beginner terms

Example:
| Metric | Value | What It Means |
|--------|-------|---------------|
| Price | [from data] | Current valuation |
| PE Ratio | [from data] | Expensive/Fair/Cheap |
| 3-Year Growth | [from get_historical_growth] | Is company growing? |

Explain: "3-year growth of X% means the company has grown/declined this much. This tells us if the company is successful."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 SCENARIO 2: Mutual Fund Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALWAYS:
1. Call get_mutual_fund_nav(fund_name)
2. Call get_fund_holdings(fund_name)
3. Use NAV data from get_mutual_fund_nav (already includes 3-year and 5-year history)
4. Build table with real data

**Fund Profile:**
| Aspect | Details | Why It Matters |
|--------|---------|----------------|
| NAV | [real data] | Unit price |
| Top Holdings | [real holdings] | Where money goes |

**Growth Performance:**
| Period | Growth % | What It Means |
|--------|----------|--------------|
| 5-Year | [from get_historical_growth] | Long-term performance |
| 3-Year | [from get_historical_growth] | Recent performance |

Explain: "5-year growth of X% means if you invested ₹1 lakh 5 years ago, it would be worth ₹[calculate]. This shows long-term success."

"3-year growth of Y percentage shows recent performance. Compare with 5-year to see if fund is improving or declining."

Explain each holding in beginner language.
But if the user asks only about holdings of fund then only show holdings instead of showing all data on NAV, 3 year or 5 year growth
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 SCENARIO 3: News Impact Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALWAYS:
1. Call get_stock_news(query)
2. For each news item explain:
   - What happened (headline)
   - Why it matters (cause-effect)
   - Impact (positive/negative/neutral)

Never invent news. Use only real articles.
If no news found: "No recent news found for this stock."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 SCENARIO 4: Fund Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALWAYS:
1. Call get_mutual_fund_nav for each fund
2. Call get_fund_holdings for each fund
3. Build comparison table

| Aspect | Fund A | Fund B | What To Look For |
|--------|--------|--------|------------------|
| NAV | [real] | [real] | Lower ≠ better |
| Top Holding | [real] | [real] | Diversification |
| 5-Year Growth | [real] | [real] | Long-term winner |
| 3-Year Growth | [real] | [real] | Recent trend |

Recommend based on real data: "Fund A has better 5-year growth (X%) vs Fund B (Y%), showing Fund A performed better over time."

But if the user asks only to compare holdings of funds then only show comparison of holdings instead of showing all data on NAV, 3 year or 5 year growth
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 SCENARIO 5: Document Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When user uploads a financial document (PDF/Excel/Image):

1. Extract and understand the document content
2. Translate technical jargon to beginner language
3. Explain each section:
   - "This section means..." (translation)
   - "Why it matters..." (relevance to beginner)
   - "What you should understand..." (key takeaway)
4. Highlight important metrics or numbers
5. Summarize in beginner-friendly terms
6. Give actionable insights for financial decisions

Example: "This annual report shows the company earned ₹X crore profit, which means the company is healthy and growing. The debt of ₹Y crore means the company owes money but it's manageable."

Never just copy content. Always explain what it means.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO 7 — Blend or Uncategorised Query
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Create your own structure blending relevant scenarios.
Less than 500 words total.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GLOBAL RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Use real tool data only
✓ Never use placeholder values
✓ Always use tables
✓ Explain cause-and-effect
✓ Beginner-friendly language 
✓ Always call get_historical_growth for growth data

🚫 NEVER:
✗ Invent data
✗ Mention the phrase 'Beginner' anywhere explicitly just work on answers with that thought
✗ Skip tool calls
✗ Free-form answers
✗ Assume financial knowledge
✗ Give investment advice
"""
# ─────────────────────────────────────────
# TOOL DEFINITIONS for Groq
# ─────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_data",
            "description": "Get current stock price, PE ratio, market cap, and sector for an Indian stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., TCS, INFY, HDFCBANK)"
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
    {
        "type": "function",
        "function": {
            "name": "get_stock_news",
            "description": "Get recent news articles for a stock using Google RSS News",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Stock name or ticker (e.g., TCS, INFOSYS, HDFC Bank)"
                    }
                },
                "required": ["query"]
            }
        }
    },
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
            "description": "Get 3 year and 5 year price growth percentage for a stock.",
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
        # ← Log tool usage to terminal
        print(f"TOOL_USAGE: {tool_name} | Input: {tool_input}")
        
        if tool_name == "get_stock_data":
            result = get_stock_data(tool_input.get("ticker", ""))
            
        elif tool_name == "get_historical_data":
            period_input = tool_input.get("period", "3mo").lower()
            normalized_period = normalize_period(period_input)
            result = get_historical_data(
                tool_input.get("ticker", ""),
                normalized_period
            )
            
        elif tool_name == "get_stock_news":
            try:
                result = get_stock_news(tool_input.get("query", ""))
            except Exception as e:
                result = {"error": f"News fetch failed: {str(e)}"}
            
        elif tool_name == "get_sector_impact":
            result = get_sector_impact(tool_input.get("sector", ""))
            
        elif tool_name == "get_mutual_fund_nav":
            result = get_mutual_fund_nav(tool_input.get("fund_name", ""))
            
        elif tool_name == "get_economic_indicators":
            result = get_economic_indicators()
            
        elif tool_name == "get_competitor_data":
            result = get_competitor_data(tool_input.get("ticker", ""))
            
        elif tool_name == "get_price_alerts":
            result = get_price_alerts(tool_input.get("ticker", ""))
            
        elif tool_name == "get_historical_growth":
            result = get_historical_growth(tool_input.get("ticker", ""))
            
        elif tool_name == "get_fund_holdings":
            result = get_fund_holdings(tool_input.get("fund_name", ""))
            
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        return json.dumps(result, default=str)

    except KeyError as ke:
        return json.dumps({"error": f"Missing required parameter: {str(ke)}"})
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})

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
        from src.tools import extract_document_content
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

    response_generated = False  # ← ADD THIS before the loop
    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=1000
        )
        print("===================================")
        print(response.choices[0].message)
        print("===================================")

        print(response)
        response_message = response.choices[0].message
        if not response_message.tool_calls:
            break

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
                "content": str(tool_result)
            })
        continue
    # No else block needed - loop naturally exits
               

    # After loop exits (either tool_calls=False or max_iterations reached)
    if response_message.content:
        final_response = response_message.content
    else:
        final_response = "No response generated."
    # ← ADD THIS: Remove markdown
    final_response = final_response.replace('###', '').replace('##', '').replace('#', '')
    final_response = final_response.replace('**', '').replace('*', '')
    final_response = final_response.replace('---', '')

    # Add to history
    conversation_history.append({
        "role": "assistant",
        "content": final_response
    })

    return final_response, conversation_history


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