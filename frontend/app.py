import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Cortex: Enterprise AI Platform", page_icon="🧠", layout="wide")


def hdr():
    return {"Authorization": f"Bearer {st.session_state['token']}"}


if "token" not in st.session_state:
    st.title("🧠 Cortex: Enterprise AI Platform")
    with st.form("login"):
        u = st.text_input("Username (admin / analyst)")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            r = requests.post(f"{API}/auth/token", data={"username": u, "password": p}, timeout=30)
            if r.ok:
                st.session_state.update(token=r.json()["access_token"], role=r.json()["role"], user=u)
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()

role = st.session_state["role"]
st.sidebar.write(f"👤 {st.session_state['user']} ({role})")
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

tabs = ["💬 Chat"] + (["📄 Documents", "📊 Dashboard", "✅ Approvals", "🧪 Evaluation"] if role == "admin" else [])
tab = dict(zip(tabs, st.tabs(tabs)))

with tab["💬 Chat"]:
    dept = st.text_input("Department filter (optional)")
    top_k = st.slider("Top-K", 1, 10, 5)
    st.session_state.setdefault("history", [])
    for m in st.session_state["history"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if q := st.chat_input("Ask about your documents…"):
        st.session_state["history"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Agents working…"):
                r = requests.post(f"{API}/query", headers=hdr(), timeout=180,
                                  json={"question": q, "top_k": top_k, "department": dept or None})
            if r.ok:
                d = r.json()
                st.markdown(d["answer"])
                st.caption(f"route={d['route']} · grounded={d['grounded']} · risk={d['risk']} · "
                           f"{d['latency_ms']} ms · {d['tokens']} tokens")
                with st.expander("Sources"):
                    for s in d["sources"]:
                        st.write(f"**{s['source']}** p.{s['page']} (score {s['score']})")
                        st.caption(s["snippet"])
                with st.expander("Agent trace"):
                    st.code("\n".join(d["trace"]))
                st.session_state["history"].append({"role": "assistant", "content": d["answer"]})
            else:
                st.error(r.json().get("detail", r.text))

if role == "admin":
    with tab["📄 Documents"]:
        f = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt", "md"])
        dep = st.text_input("Department", "general")
        if f and st.button("Ingest"):
            with st.spinner("Chunking, embedding, upserting to Pinecone…"):
                r = requests.post(f"{API}/ingest", headers=hdr(), data={"department": dep},
                                  files={"file": (f.name, f.getvalue())}, timeout=600)
            st.success(r.json()) if r.ok else st.error(r.text)

    with tab["📊 Dashboard"]:
        r = requests.get(f"{API}/audit", headers=hdr(), timeout=30)
        df = pd.DataFrame(r.json()) if r.ok else pd.DataFrame()
        q = df[df["latency_ms"] > 0] if not df.empty else df
        if q.empty:
            st.info("No queries yet.")
        else:
            c = st.columns(5)
            c[0].metric("Queries", len(q))
            c[1].metric("Avg latency (ms)", int(q["latency_ms"].mean()))
            c[2].metric("Tokens", int(q["tokens"].sum()))
            c[3].metric("Ungrounded %", f"{100 * (~q['grounded']).mean():.1f}")
            c[4].metric("Blocked/Error", int(df["status"].isin(["blocked", "error"]).sum()))
            st.line_chart(q.set_index("ts")[["latency_ms"]])
            st.bar_chart(q["route"].value_counts())
            st.dataframe(df.drop(columns=["answer"]), use_container_width=True)

    with tab["✅ Approvals"]:
        r = requests.get(f"{API}/approvals", headers=hdr(), timeout=30)
        for row in (r.json() if r.ok else []):
            st.write(f"**#{row['id']}** {row['user']} · risk {row['risk']}")
            st.write(row["question"])
            st.code(row["answer"])
            a, b = st.columns(2)
            if a.button("Approve", key=f"a{row['id']}"):
                requests.post(f"{API}/approvals/{row['id']}/approve", headers=hdr(), timeout=30)
                st.rerun()
            if b.button("Reject", key=f"r{row['id']}"):
                requests.post(f"{API}/approvals/{row['id']}/reject", headers=hdr(), timeout=30)
                st.rerun()

    with tab["🧪 Evaluation"]:
        st.caption('JSON list: [{"question": "...", "relevant_sources": ["file.pdf"]}]')
        up = st.file_uploader("Eval dataset", type=["json"])
        if up and st.button("Run evaluation"):
            with st.spinner("Evaluating…"):
                import json
                r = requests.post(f"{API}/eval", headers=hdr(), json=json.load(up), timeout=1800)
            if r.ok:
                st.json(r.json()["summary"])
                st.dataframe(pd.DataFrame(r.json()["rows"]))
            else:
                st.error(r.text)
