# api.py — FastAPI backend

import sys
import os
import json
import datetime
from collections import Counter

# Add src to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tempfile
from agent import run_agent
from tools import (
    get_stock_data, get_historical_data, get_stock_news, get_sector_impact,
    get_mutual_fund_nav, get_economic_indicators, get_competitor_data,
    get_price_alerts, get_historical_growth, get_fund_holdings
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

conversation_store = {}

# ← NEW: Storage for feedback and tool tracking
feedback_log = []
tool_usage_log = []

# ← NEW: Add this line
user_queries_log = []

# Get absolute path to frontend folder
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.post("/chat")
async def chat(
    message: str = Form(...),
    session_id: str = Form(default="default"),
    file: UploadFile = File(default=None)
):
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    history = conversation_store[session_id]
    # ← NEW: Log the user query
    user_queries_log.append({
        "session_id": session_id,
        "message": message,
        "timestamp": datetime.datetime.now().isoformat()
    })

    pdf_path = None
    if file:  # ← CHANGED: Accept any file
        # Allowed extensions
        allowed_ext = [".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".txt"]  # ← NEW
        file_ext = f".{file.filename.split('.')[-1].lower()}"  # ← NEW: Extract extension
        
        if file_ext in allowed_ext:  # ← NEW: Check if allowed
            # Save with proper extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:  # ← CHANGED
                content = await file.read()
                tmp.write(content)
                pdf_path = tmp.name
                print(f"DEBUG: File saved to: {pdf_path}")  # ← NEW: Debug log
        else:
            return {"response": f"File type not supported. Allowed: {', '.join(allowed_ext)}", "session_id": session_id}

    response, updated_history = run_agent(message, history, pdf_path)
    conversation_store[session_id] = updated_history

    # ← NEW: Parse tool usage from agent execution and log it
    import re
    tool_pattern = r'TOOL_USAGE: (\w+)'
    tools_used = re.findall(tool_pattern, str(response))
    for tool in tools_used:
        tool_usage_log.append({
            "tool": tool,
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat()
        })
        print(f"TRACKED: {tool}")

    if pdf_path and os.path.exists(pdf_path):
        os.unlink(pdf_path)
        print(f"DEBUG: Temp file deleted: {pdf_path}")  # ← NEW: Debug log

    return {"response": response, "session_id": session_id}

@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    if session_id in conversation_store:
        del conversation_store[session_id]
    return {"message": "Conversation cleared"}
# Add ticker for chart - NEW
@app.get("/chart-data/{ticker}")
async def get_chart_data(
    ticker: str,
    period: str = "6mo"
):
    try:
        from technical_analysis import get_technical_indicators

        result = get_technical_indicators(
            f"{ticker}.NS",
            period
        )

        print("\n========== RESPONSE ==========")
        print(result)
        print("==============================\n")

        return result

    except Exception as e:
        print(e)
        return {"error": str(e)}
# ═══════════════════════════════════════════════════════════════
# ← NEW: FEEDBACK TRACKING (Point 3)
# ═══════════════════════════════════════════════════════════════

@app.post("/feedback")
async def record_feedback(
    message_id: str = Form(...),
    feedback: str = Form(...),  # "helpful" or "not_helpful"
    session_id: str = Form(...),
    timestamp: str = Form(...)
):
    """Record user feedback on responses"""
    feedback_record = {
        "message_id": message_id,
        "feedback": feedback,
        "session_id": session_id,
        "timestamp": timestamp,
        "recorded_at": datetime.datetime.now().isoformat()
    }
    
    feedback_log.append(feedback_record)
    print(f"FEEDBACK: {feedback} | Message: {message_id} | Session: {session_id}")
    
    return {"status": "feedback_recorded"}


@app.get("/analytics/feedback")
async def get_feedback_analytics():
    """Get feedback statistics"""
    if not feedback_log:
        return {"message": "No feedback yet"}
    
    helpful_count = sum(1 for f in feedback_log if f["feedback"] == "helpful")
    not_helpful_count = sum(1 for f in feedback_log if f["feedback"] == "not_helpful")
    total = len(feedback_log)
    
    return {
        "total_feedback": total,
        "helpful": helpful_count,
        "not_helpful": not_helpful_count,
        "satisfaction_rate": round((helpful_count / total * 100), 2) if total > 0 else 0,
        "recent_feedback": feedback_log[-10:]  # Last 10 feedback items
    }


# ═══════════════════════════════════════════════════════════════
# ← NEW: TOOL TRACKING (Point 2)
# ═══════════════════════════════════════════════════════════════

@app.post("/track-tool")
async def track_tool(
    tool_name: str = Form(...),
    session_id: str = Form(...)
):
    """Track which tools are being used"""
    tool_usage_log.append({
        "tool": tool_name,
        "session_id": session_id,
        "timestamp": datetime.datetime.now().isoformat()
    })
    print(f"TOOL_TRACKED: {tool_name} | Session: {session_id}")
    return {"status": "tracked"}


@app.get("/analytics/tools")
async def get_tool_analytics():
    """Get tool usage statistics"""
    if not tool_usage_log:
        return {"message": "No tool usage data yet"}
    
    tool_counts = Counter([log["tool"] for log in tool_usage_log])
    
    return {
        "total_tool_calls": len(tool_usage_log),
        "tool_breakdown": dict(tool_counts),
        "most_used": [{"tool": tool, "count": count} for tool, count in tool_counts.most_common(5)]
    }


# ═══════════════════════════════════════════════════════════════
# ← NEW: DATA EXPORT (for analysis)
# ═══════════════════════════════════════════════════════════════

@app.get("/export/feedback-csv")
async def export_feedback_csv():
    """Export feedback as CSV for analysis"""
    if not feedback_log:
        return {"message": "No feedback to export"}
    
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=["message_id", "feedback", "session_id", "timestamp", "recorded_at"])
    writer.writeheader()
    writer.writerows(feedback_log)
    
    return {
        "csv": output.getvalue(),
        "total_records": len(feedback_log)
    }


@app.get("/export/tool-usage-csv")
async def export_tool_usage_csv():
    """Export tool usage as CSV for analysis"""
    if not tool_usage_log:
        return {"message": "No tool usage data to export"}
    
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=["tool", "session_id", "timestamp"])
    writer.writeheader()
    writer.writerows(tool_usage_log)
    
    return {
        "csv": output.getvalue(),
        "total_records": len(tool_usage_log)
    }

# ═══════════════════════════════════════════════════════════════
# ← NEW: COMBINED ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════

@app.get("/analytics/dashboard")
async def get_dashboard():
    """Get complete analytics dashboard"""
    
    # Feedback stats
    if feedback_log:
        helpful = sum(1 for f in feedback_log if f["feedback"] == "helpful")
        satisfaction = round((helpful / len(feedback_log) * 100), 2)
    else:
        helpful = 0
        satisfaction = 0
    
    # Tool stats
    if tool_usage_log:
        tool_counts = Counter([log["tool"] for log in tool_usage_log])
        most_used_tools = dict(tool_counts.most_common(5))
    else:
        most_used_tools = {}
    
    return {
        "feedback": {
            "total": len(feedback_log),
            "helpful": helpful,
            "not_helpful": len(feedback_log) - helpful,
            "satisfaction_rate": satisfaction
        },
        "tools": {
            "total_calls": len(tool_usage_log),
            "most_used": most_used_tools
        },
        "sessions": len(conversation_store)
    }

# ← NEW: Get user queries
@app.get("/analytics/user-queries")
async def get_user_queries():
    """Get all user queries"""
    if not user_queries_log:
        return {"message": "No queries yet"}
    
    return {
        "total_queries": len(user_queries_log),
        "recent_queries": user_queries_log[-50:],  # Last 50 queries
        "all_queries": user_queries_log
    }


# ← NEW: Export user queries as CSV
@app.get("/export/user-queries-csv")
async def export_queries_csv():
    """Export user queries as CSV"""
    if not user_queries_log:
        return {"message": "No queries to export"}
    
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=["timestamp", "session_id", "message"])
    writer.writeheader()
    writer.writerows(user_queries_log)
    
    return {
        "csv": output.getvalue(),
        "total_records": len(user_queries_log)
    }
# ═══════════════════════════════════════════════════════════════
# ← NEW: CLEAR ANALYTICS (for testing/reset)
# ═══════════════════════════════════════════════════════════════

@app.delete("/analytics/clear")
async def clear_analytics():
    """Clear all feedback and tool tracking data"""
    global feedback_log, tool_usage_log
    
    feedback_count = len(feedback_log)
    tool_count = len(tool_usage_log)
    
    feedback_log = []
    tool_usage_log = []
    
    return {
        "message": "Analytics cleared",
        "feedback_records_deleted": feedback_count,
        "tool_records_deleted": tool_count
    }