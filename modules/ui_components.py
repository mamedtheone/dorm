"""
ui_components.py — Dorm-Net Edizione Italiana
==============================================
Aesthetic: Italian editorial luxury meets academic dark tool.
Fonts: Cormorant Garamond (display) · Libre Baskerville (body) · DM Mono (labels/code)
Color: Deep navy/charcoal · Gold accents · Soft cream text
Texture: Grain overlay · Atmospheric radial gradients
"""

from __future__ import annotations
from typing import Callable
import streamlit as st
from modules.persona_module import PERSONAS

PERSONA_META: dict[str, dict] = {
    "software": {"name": "Prof. Alan Turing", "dept": "Dept. of Software Engineering", "sigil": "SWE", "tags": ["Systems", "Algorithms"], "color": "#c9a84c"},
    "mechanical": {"name": "Prof. L. da Vinci", "dept": "Dept. of Mechanical Engineering", "sigil": "ME", "tags": ["Mechanics", "Fluids"], "color": "#d4854a"},
    "electrical": {"name": "Prof. Nikola Tesla", "dept": "Dept. of Electrical Engineering", "sigil": "EE", "tags": ["Circuits", "Signals"], "color": "#8b7cf8"},
    "math": {"name": "Prof. Emmy Noether", "dept": "Dept. of Applied Mathematics", "sigil": "MTH", "tags": ["Calculus", "Proofs"], "color": "#5fa8d3"},
    "eli12": {"name": "Prof. Carl Sagan", "dept": "Foundations & Intuition", "sigil": "ELI", "tags": ["Analogies", "Clarity"], "color": "#c9a84c"},
}

MODE_META: dict[str, dict] = {
    "answer": {"label": "Q&A Answer", "icon": "▸"},
    "concept_breakdown": {"label": "Concept Map", "icon": "⊞"},
    "diagnosis": {"label": "Error Diagnosis", "icon": "⊘"},
    "notes": {"label": "Study Notes", "icon": "≡"},
}

NAV_ITEMS = [("chat","◈","Chat"),("quiz","⊟","Quiz"),("ocr","⊡","OCR"),("health","◎","System")]

DARK_THEME_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400;1,500;1,600&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Instrument+Serif:ital@0;1&display=swap');
:root{--ink:#0e0e14;--ink-deep:#07070c;--panel:#131320;--panel-lite:#1a1a2e;--lift:#20203a;--lift2:#282848;--gold:#c9a84c;--gold-light:#e8c97a;--gold-dim:rgba(201,168,76,0.12);--gold-str:rgba(201,168,76,0.30);--gold-glow:rgba(201,168,76,0.18);--cream:#ede8df;--cream-dim:rgba(237,232,223,0.5);--violet:#8b7cf8;--ok:#34c98a;--err:#f05252;--t1:#ede8df;--t2:#b8b0a2;--t3:#6e6860;--t4:#3a3830;--border:rgba(201,168,76,0.14);--border-2:rgba(237,232,223,0.07);--serif-d:'Cormorant Garamond','EB Garamond',Georgia,serif;--serif-b:'Libre Baskerville',Georgia,serif;--mono:'DM Mono','JetBrains Mono',monospace;--inst:'Instrument Serif',serif}
html,body,[class*="css"],.stApp{font-family:var(--serif-b)!important;background:var(--ink)!important;color:var(--t1)!important;-webkit-font-smoothing:antialiased!important}
.block-container{padding-top:0!important;max-width:100%!important;padding-left:0!important;padding-right:0!important}
#MainMenu,footer,header{visibility:hidden!important}
::-webkit-scrollbar{width:3px;height:3px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--lift2);border-radius:2px}
[data-testid="stSidebar"]{background:var(--panel)!important;border-right:1px solid var(--border)!important}
[data-testid="stSidebar"]>div:first-child{padding:0!important}
[data-testid="stSidebar"] label{font-family:var(--mono)!important;font-size:0.6rem!important;letter-spacing:0.2em!important;text-transform:uppercase!important;color:var(--t3)!important}/* Add this to your DARK_THEME_CSS string */
.nav-label-text { 
    white-space: nowrap !important;
    word-break: normal !important;
    overflow-wrap: normal !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    font-size: 0.6rem !important;
    width: 100% !important;
}
[data-testid="stSidebar"] [data-baseweb="select"]>div{background:var(--lift)!important;border:1px solid var(--border-2)!important;border-radius:3px!important;color:var(--t1)!important;font-family:var(--serif-b)!important;font-size:0.82rem!important}
[data-testid="stSidebar"] [data-baseweb="select"]>div:hover{border-color:var(--gold-str)!important}
[data-testid="stSidebar"] [data-baseweb="select"]>div:focus-within{border-color:var(--gold)!important;box-shadow:0 0 0 2px var(--gold-dim)!important}
[data-testid="stSidebar"] [data-testid="stFileUploader"]{background:var(--lift)!important;border:1px dashed var(--border-2)!important;border-radius:3px!important}
[data-testid="stSidebar"] .stToggle p{font-family:var(--serif-b)!important;font-size:0.78rem!important;color:var(--t2)!important}
.sb-brand-wrap{padding:2.2rem 1.6rem 1.6rem;border-bottom:1px solid var(--border)}
.sb-eyebrow{font-family:var(--mono);font-size:0.57rem;letter-spacing:0.28em;text-transform:uppercase;color:var(--gold);opacity:.75;margin-bottom:.55rem}
.sb-name{font-family:var(--serif-d);font-size:2.2rem;font-weight:600;font-style:italic;line-height:1;letter-spacing:-.01em;background:linear-gradient(135deg,#ede8df 30%,#c9a84c 52%,#e8c97a 70%,#ede8df 90%);background-size:250% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:shimmer 7s linear infinite}
@keyframes shimmer{0%{background-position:250% center}100%{background-position:-250% center}}
.sb-tagline{font-family:var(--mono);font-size:0.58rem;color:var(--t3);letter-spacing:.12em;text-transform:uppercase;margin-top:.5rem}
.prof-card{margin:1.2rem 1.2rem 0;background:linear-gradient(135deg,var(--panel-lite) 0%,var(--lift) 100%);border:1px solid var(--border);border-radius:4px;padding:1rem 1.1rem;position:relative;overflow:hidden}
.prof-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--gold),transparent);opacity:.7}
.prof-card::after{content:'';position:absolute;bottom:-40px;right:-40px;width:120px;height:120px;background:radial-gradient(circle,var(--gold-glow) 0%,transparent 70%);pointer-events:none}
.prof-eyebrow{font-family:var(--mono);font-size:.56rem;letter-spacing:.2em;text-transform:uppercase;color:var(--gold);margin-bottom:.5rem}
.prof-name{font-family:var(--serif-d);font-size:1.2rem;font-weight:600;font-style:italic;color:var(--cream);line-height:1.2;margin-bottom:.25rem}
.prof-dept{font-family:var(--serif-b);font-size:.7rem;color:var(--t2);font-style:italic;margin-bottom:.65rem}
.prof-tag{display:inline-block;font-family:var(--mono);font-size:.54rem;letter-spacing:.12em;text-transform:uppercase;color:var(--gold);background:var(--gold-dim);border:1px solid rgba(201,168,76,.25);border-radius:2px;padding:2px 7px;margin-right:4px}
.kernel-row{display:flex;align-items:center;gap:8px;margin:1rem 1.2rem 0;padding:.55rem .85rem;border-radius:3px}
.kernel-row.online{background:rgba(52,201,138,.07);border:1px solid rgba(52,201,138,.18)}
.kernel-row.offline{background:rgba(240,82,82,.07);border:1px solid rgba(240,82,82,.18)}
.kernel-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.kernel-dot.on{background:var(--ok);animation:kp 2.5s ease-in-out infinite}
.kernel-dot.off{background:var(--err)}
@keyframes kp{0%,100%{opacity:1}50%{opacity:.35}}
.kernel-label{font-family:var(--mono);font-size:.6rem;letter-spacing:.05em}
.kernel-label.on{color:var(--ok)}.kernel-label.off{color:var(--err)}
.sb-section{padding:0 1.2rem;margin-top:1rem}
.sb-sec-title{font-family:var(--mono);font-size:.56rem;letter-spacing:.22em;text-transform:uppercase;color:var(--t3);margin-bottom:.5rem;display:flex;align-items:center;gap:8px}
.sb-sec-title::after{content:'';flex:1;height:1px;background:var(--border-2)}
.doc-badge{display:flex;align-items:center;gap:7px;padding:.4rem .6rem;background:var(--lift);border:1px solid var(--border-2);border-radius:2px;margin-bottom:3px;font-family:var(--mono);font-size:.6rem;color:var(--t2);overflow:hidden}
.doc-badge-icon{color:var(--gold);flex-shrink:0}
.doc-badge-text{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chunk-badge{font-family:var(--mono);font-size:.6rem;color:var(--ok);background:rgba(52,201,138,.08);border:1px solid rgba(52,201,138,.2);border-radius:2px;padding:2px 8px;display:inline-block;margin-top:4px}
.sb-bottom{margin-top:auto;padding:1.2rem;border-top:1px solid var(--border-2)}
.sb-bottom-key{font-family:var(--mono);font-size:.56rem;letter-spacing:.1em;color:var(--t3);text-transform:uppercase;margin-bottom:2px}
.sb-bottom-val{font-family:var(--mono);font-size:.65rem;color:var(--violet)}
.page-header{padding:2.8rem 3rem 1.8rem;border-bottom:1px solid var(--border-2);background:linear-gradient(180deg,rgba(10,10,22,.9) 0%,transparent 100%);position:relative;overflow:hidden}
.page-header::after{content:'';position:absolute;top:-80px;right:-80px;width:340px;height:340px;background:radial-gradient(circle,var(--gold-glow) 0%,transparent 65%);pointer-events:none}
.ph-eyebrow{font-family:var(--mono);font-size:.6rem;letter-spacing:.3em;text-transform:uppercase;color:var(--gold);margin-bottom:1rem;opacity:.7}
.ph-title{font-family:var(--serif-d);font-size:3.2rem;font-weight:300;line-height:1.05;letter-spacing:-.02em;color:var(--cream);margin-bottom:.5rem}
.ph-title em{font-style:italic;font-weight:700;background:linear-gradient(135deg,var(--cream) 0%,var(--gold) 45%,var(--gold-light) 65%,var(--cream) 100%);background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:shimmer 6s linear infinite}
.ph-sub{font-family:var(--inst);font-size:1.05rem;font-style:italic;color:var(--t2)}
.orn-rule{display:flex;align-items:center;gap:1rem;margin-top:1.5rem}
.orn-rule::before,.orn-rule::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,transparent,var(--gold),transparent);opacity:.25}
.orn-diamonds{font-size:.42rem;color:var(--gold);opacity:.5;letter-spacing:8px}
.mode-bar{display:flex;align-items:center;gap:.35rem;padding:.85rem 3rem;border-bottom:1px solid var(--border-2)}
.mode-chip{font-family:var(--mono);font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);background:transparent;border:1px solid transparent;border-radius:2px;padding:.35rem .85rem;cursor:default;transition:all .15s}
.mode-chip.active{color:var(--gold);background:var(--gold-dim);border-color:rgba(201,168,76,.28)}
.inline-prof-card{display:flex;align-items:center;gap:1.1rem;padding:.9rem 1.2rem;background:linear-gradient(135deg,var(--panel-lite) 0%,var(--lift) 100%);border:1px solid var(--border);border-radius:4px;margin-bottom:1.5rem;position:relative;overflow:hidden}
.inline-prof-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--gold),transparent);opacity:.5}
.ipc-sigil{font-family:var(--serif-d);font-size:1.8rem;font-weight:700;font-style:italic;line-height:1;flex-shrink:0;width:50px;text-align:center}
.ipc-body{flex:1;min-width:0}
.ipc-name{font-family:var(--serif-d);font-size:1.05rem;font-weight:600;font-style:italic;color:var(--cream);line-height:1.2;margin-bottom:.2rem}
.ipc-dept{font-family:var(--mono);font-size:.58rem;color:var(--t3);letter-spacing:.08em}
.ipc-right{display:flex;flex-direction:column;align-items:flex-end;gap:4px;flex-shrink:0}
.ipc-mode-icon{font-family:var(--mono);font-size:1.1rem;color:var(--t3)}
.ipc-mode-label{font-family:var(--mono);font-size:.55rem;letter-spacing:.12em;text-transform:uppercase;color:var(--t3)}
.ipc-level{font-family:var(--mono);font-size:.55rem;color:var(--gold);background:var(--gold-dim);border:1px solid rgba(201,168,76,.2);border-radius:2px;padding:1px 6px}
[data-testid="stChatMessage"]{background:transparent!important;border:none!important;padding:0!important}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){border-right:2px solid rgba(201,168,76,.35)!important;padding-right:12px!important;margin-right:-14px!important}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]){background:linear-gradient(135deg,var(--panel-lite) 0%,rgba(26,26,46,.6) 100%)!important;border:1px solid var(--border)!important;border-left:2px solid var(--gold)!important;border-radius:0 4px 4px 0!important;padding:1.2rem 1.4rem!important;margin-bottom:1rem!important;position:relative!important}
[data-testid="stChatMessage"] p{font-family:var(--serif-b)!important;font-size:.9rem!important;line-height:1.85!important;color:var(--t1)!important}
[data-testid="stChatMessage"] h1,[data-testid="stChatMessage"] h2,[data-testid="stChatMessage"] h3{font-family:var(--serif-d)!important;font-style:italic!important;font-weight:600!important;color:var(--cream)!important;border-bottom:1px solid var(--border-2)!important;padding-bottom:.35rem!important;margin:1.2rem 0 .6rem!important}
[data-testid="stChatMessage"] code{font-family:var(--mono)!important;font-size:.82em!important;background:var(--lift2)!important;border:1px solid var(--border-2)!important;border-radius:2px!important;padding:.1em .4em!important;color:#a5c9ff!important}
[data-testid="stChatMessage"] pre{background:var(--ink-deep)!important;border:1px solid var(--border-2)!important;border-left:2px solid var(--gold)!important;border-radius:0 3px 3px 0!important;padding:1rem 1.2rem!important;margin:.8rem 0!important;box-shadow:0 4px 24px rgba(0,0,0,.3)!important}
[data-testid="stChatMessage"] pre code{background:transparent!important;border:none!important;padding:0!important;color:#c9deff!important;font-size:.8rem!important;line-height:1.65!important}
[data-testid="stChatMessage"] blockquote{border-left:2px solid var(--gold)!important;background:var(--panel-lite)!important;margin:.8rem 0!important;padding:.75rem 1.1rem!important;border-radius:0 3px 3px 0!important;font-family:var(--serif-d)!important;font-style:italic!important;font-size:1.05rem!important;color:var(--t2)!important}
[data-testid="stChatMessage"] table{border-collapse:collapse!important;width:100%!important;font-size:.82rem!important}
[data-testid="stChatMessage"] th{background:var(--lift)!important;border:1px solid var(--border-2)!important;padding:5px 10px!important;font-family:var(--mono)!important;font-size:.6rem!important;letter-spacing:.1em!important;text-transform:uppercase!important;color:var(--t2)!important}
[data-testid="stChatMessage"] td{border:1px solid var(--border-2)!important;padding:5px 10px!important;font-family:var(--serif-b)!important;color:var(--t1)!important}
[data-testid="stChatInput"]{background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:4px!important;box-shadow:0 -4px 24px rgba(0,0,0,.3)!important}
[data-testid="stChatInput"]:focus-within{border-color:rgba(201,168,76,.45)!important;box-shadow:0 0 0 3px var(--gold-dim),0 -4px 24px rgba(0,0,0,.4)!important}
[data-testid="stChatInput"] textarea{font-family:var(--serif-b)!important;font-size:.9rem!important;color:var(--t1)!important;background:transparent!important}
[data-testid="stChatInput"] textarea::placeholder{color:var(--t4)!important;font-style:italic!important}
.sources-wrap{margin-top:1rem;padding-top:.9rem;border-top:1px solid var(--border-2)}
.sources-label{font-family:var(--mono);font-size:.56rem;letter-spacing:.2em;text-transform:uppercase;color:var(--t3);margin-bottom:.6rem}
.src-card{display:flex;align-items:flex-start;gap:.75rem;padding:.55rem .8rem;background:var(--lift);border:1px solid var(--border-2);border-radius:3px;margin-bottom:.4rem;transition:border-color .15s,background .15s}
.src-card:hover{border-color:rgba(201,168,76,.25);background:var(--lift2)}
.src-idx{font-family:var(--mono);font-size:.62rem;font-weight:500;color:var(--gold);flex-shrink:0;margin-top:1px;min-width:1.2rem}
.src-body{flex:1;min-width:0}
.src-title{font-family:var(--serif-b);font-size:.76rem;font-weight:700;color:var(--t1);margin-bottom:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.src-meta{font-family:var(--mono);font-size:.57rem;color:var(--t3);letter-spacing:.05em}
.src-snippet{font-family:var(--serif-b);font-size:.72rem;font-style:italic;color:var(--t2);line-height:1.5;margin-top:4px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.src-score{font-family:var(--mono);font-size:.56rem;color:var(--ok);background:rgba(52,201,138,.08);border:1px solid rgba(52,201,138,.18);border-radius:2px;padding:1px 5px;flex-shrink:0;align-self:flex-start}
.fups-wrap{margin-top:1rem;padding-top:.9rem;border-top:1px solid var(--border-2)}
.fups-label{font-family:var(--inst);font-size:.78rem;font-style:italic;color:var(--t3);margin-bottom:.6rem}
.fup-item{display:flex;align-items:flex-start;gap:.65rem;padding:.45rem .75rem;background:var(--panel-lite);border:1px solid var(--border-2);border-radius:2px;margin-bottom:.35rem;font-family:var(--serif-b);font-size:.76rem;font-style:italic;color:var(--t2);transition:border-color .15s,color .15s}
.fup-item:hover{border-color:rgba(201,168,76,.28);color:var(--cream)}
.fup-num{font-family:var(--mono);font-size:.6rem;color:var(--gold);font-weight:500;flex-shrink:0;margin-top:1px;font-style:normal}
.quiz-card{background:var(--panel-lite);border:1px solid var(--border);border-radius:4px;padding:1.1rem 1.2rem;margin-bottom:1rem;position:relative;overflow:hidden}
.quiz-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--gold),transparent);opacity:.4}
.quiz-q-num{font-family:var(--mono);font-size:.57rem;color:var(--gold);letter-spacing:.12em;font-weight:500;margin-bottom:.45rem}
.quiz-q-text{font-family:var(--serif-b);font-size:.88rem;font-weight:700;color:var(--t1);line-height:1.5;margin-bottom:.75rem}
.quiz-opt{display:flex;align-items:flex-start;gap:.65rem;padding:.45rem .75rem;border-radius:2px;border:1px solid var(--border-2);margin-bottom:.35rem;font-family:var(--serif-b);font-size:.8rem;color:var(--t2);background:var(--lift);line-height:1.45}
.quiz-opt.correct{border-color:rgba(52,201,138,.4);background:rgba(52,201,138,.06);color:var(--t1)}
.quiz-opt-lbl{font-family:var(--mono);font-size:.6rem;font-weight:700;flex-shrink:0;margin-top:2px}
.quiz-opt.correct .quiz-opt-lbl{color:var(--ok)}.quiz-opt:not(.correct) .quiz-opt-lbl{color:var(--t3)}
.quiz-explanation{margin-top:.65rem;padding:.5rem .8rem;background:var(--ink-deep);border-left:2px solid var(--ok);border-radius:0 2px 2px 0;font-family:var(--serif-b);font-size:.72rem;font-style:italic;color:var(--t2);line-height:1.55}
.health-grid{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin-bottom:1rem}
.health-cell{background:var(--panel-lite);border:1px solid var(--border);border-radius:3px;padding:.9rem 1rem}
.health-val{font-family:var(--serif-d);font-size:1.6rem;font-weight:600;font-style:italic;color:var(--cream);line-height:1;margin-bottom:4px}
.health-lbl{font-family:var(--mono);font-size:.56rem;letter-spacing:.12em;text-transform:uppercase;color:var(--t3)}
.health-stack{background:var(--ink-deep);border:1px solid var(--border-2);border-radius:3px;overflow:hidden}
.health-row-item{display:flex;justify-content:space-between;align-items:center;padding:.45rem .9rem;border-bottom:1px solid var(--border-2);font-family:var(--mono);font-size:.6rem}
.health-row-item:last-child{border-bottom:none}
.health-row-key{color:var(--t3);letter-spacing:.08em}.health-row-val{color:var(--t1)}
.ocr-meta-row{display:flex;gap:5px;margin:8px 0;flex-wrap:wrap}
.sect-divider{display:flex;align-items:center;gap:1rem;margin:.75rem 0}
.sect-divider-line{flex:1;height:1px;background:var(--border-2)}
.sect-divider-text{font-family:var(--inst);font-size:.7rem;font-style:italic;color:var(--t3);white-space:nowrap}
.stButton>button{
    font-family:var(--mono)!important;
    font-size:.55rem!important;   /* slightly smaller */
    letter-spacing:.08em!important; /* reduce spacing */
    text-transform:uppercase!important;

    white-space: nowrap !important;        /* 🔥 CRITICAL */
    word-break: normal !important;         /* 🔥 CRITICAL */
    overflow-wrap: normal !important;      /* 🔥 CRITICAL */

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    min-width: 80px !important;   /* 🔥 gives text space */
    padding:6px 10px!important;

    border-radius:2px!important;
    border:1px solid var(--border-2)!important;
    background:var(--lift)!important;
    color:var(--t2)!important;
    transition:all .15s!important;
    height:auto!important;
}
.stButton>button:hover{border-color:var(--gold-str)!important;color:var(--gold)!important;background:var(--gold-dim)!important}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{font-family:var(--serif-b)!important;font-size:.85rem!important;background:var(--lift)!important;border:1px solid var(--border-2)!important;border-radius:2px!important;color:var(--t1)!important}
[data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus{border-color:rgba(201,168,76,.4)!important;box-shadow:0 0 0 2px var(--gold-dim)!important}
[data-testid="stExpander"]{background:var(--panel-lite)!important;border:1px solid var(--border)!important;border-radius:3px!important}
[data-testid="stExpander"] summary{font-family:var(--mono)!important;font-size:.6rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;color:var(--t2)!important}
[data-testid="stAlert"]{border-radius:3px!important;font-family:var(--serif-b)!important;font-size:.82rem!important}
[data-testid="column"]>div:first-child{padding:0!important}
</style>"""


def inject_css():
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "messages":[],"selected_model":"mistral:latest","selected_persona":"software",
        "subject_hint":"engineering","step_by_step":True,"debug_mode":False,
        "user_level":"auto","current_mode":"answer","indexed_docs":[],
        "ocr_result":None,"use_ocr_in_next_query":False,"is_processing":False,
        "ollama_online":False,"db_chunk_count":0,"last_rag_sources":[],
        "last_followups":[],"last_detected_level":"basic","last_quiz":[],
        "last_notes":"","active_panel":"chat","_pending_input":None,
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar(available_models, on_pdf_upload, on_clear_chat):
    with st.sidebar:
        st.markdown('<div class="sb-brand-wrap"><div class="sb-eyebrow">Engineering Tutor · Offline</div><div class="sb-name">Dorm·Net</div><div class="sb-tagline">Grounded · Local · Private</div></div>', unsafe_allow_html=True)
        _render_sb_prof_card()
        online = st.session_state.get("ollama_online", False)
        model  = st.session_state.get("selected_model","n/a")
        cls = "online" if online else "offline"
        dot_c = "on" if online else "off"
        lbl_c = "on" if online else "off"
        lbl = f"Ready · {model}" if online else "Kernel offline"
        st.markdown(f'<div class="kernel-row {cls}"><div class="kernel-dot {dot_c}"></div><span class="kernel-label {lbl_c}">{lbl}</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section"><div class="sb-sec-title">Model</div>', unsafe_allow_html=True)
        model_opts = available_models or ["mistral:latest","phi3:latest"]
        midx = model_opts.index(st.session_state.selected_model) if st.session_state.selected_model in model_opts else 0
        st.session_state.selected_model = st.selectbox("Model", model_opts, index=midx, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section"><div class="sb-sec-title">Professor</div>', unsafe_allow_html=True)
        p_keys = list(PERSONAS.keys())
        pidx = p_keys.index(st.session_state.selected_persona) if st.session_state.selected_persona in p_keys else 0
        st.session_state.selected_persona = st.selectbox("Persona", p_keys, index=pidx, format_func=lambda k: f"{PERSONA_META[k]['sigil']}  {PERSONAS[k].title}", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section"><div class="sb-sec-title">Tutor Mode</div>', unsafe_allow_html=True)
        st.session_state.current_mode = st.selectbox("Mode", list(MODE_META.keys()), format_func=lambda k: f"{MODE_META[k]['icon']}  {MODE_META[k]['label']}", index=list(MODE_META.keys()).index(st.session_state.current_mode), label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section"><div class="sb-sec-title">Config</div>', unsafe_allow_html=True)
        subj_opts = ["engineering","electrical","mechanical","software","mathematics","physics"]
        subj_cur = st.session_state.subject_hint if st.session_state.subject_hint in subj_opts else "engineering"
        st.session_state.subject_hint = st.selectbox("Subject", subj_opts, index=subj_opts.index(subj_cur))
        st.session_state.user_level = st.selectbox("Level", ["auto","basic","intermediate"], index=["auto","basic","intermediate"].index(st.session_state.user_level))
        col_a,col_b = st.columns(2)
        with col_a: st.session_state.step_by_step = st.toggle("Steps", value=st.session_state.step_by_step)
        with col_b: st.session_state.debug_mode = st.toggle("Debug", value=st.session_state.debug_mode)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section"><div class="sb-sec-title">Knowledge Base</div>', unsafe_allow_html=True)
        pdf = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
        if pdf and st.button("Index PDF", use_container_width=True): on_pdf_upload(pdf)
        db_chunks = st.session_state.get("db_chunk_count",0)
        if db_chunks: st.markdown(f'<div class="chunk-badge">⊕ {db_chunks:,} vectors indexed</div>', unsafe_allow_html=True)
        for doc in st.session_state.get("indexed_docs",[]):
            src = doc.get("source","?")
            st.markdown(f'<div class="doc-badge"><span class="doc-badge-icon">◈</span><span class="doc-badge-text">{src}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section"><div class="sb-sec-title">Session</div>', unsafe_allow_html=True)
        if st.button("Clear Chat", use_container_width=True): on_clear_chat()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-bottom"><div class="sb-bottom-key">Active stack</div><div class="sb-bottom-val">ChromaDB · BM25+Vector RRF</div></div>', unsafe_allow_html=True)


def _render_sb_prof_card():
    key = st.session_state.get("selected_persona","software")
    pmeta = PERSONA_META.get(key, PERSONA_META["software"])
    persona = PERSONAS.get(key, PERSONAS["software"])
    tags_html = "".join(f'<span class="prof-tag">{t}</span>' for t in pmeta["tags"][:2])
    st.markdown(f'<div class="prof-card" style="border-color:{pmeta["color"]}33;"><div class="prof-eyebrow">Active Professor</div><div class="prof-name" style="-webkit-text-fill-color:{pmeta["color"]};color:{pmeta["color"]};">{pmeta["name"]}</div><div class="prof-dept">{pmeta["dept"]}</div><div>{tags_html}</div></div>', unsafe_allow_html=True)


def render_page_header():
    key = st.session_state.get("selected_persona","software")
    persona = PERSONAS.get(key, PERSONAS["software"])
    mode = MODE_META.get(st.session_state.get("current_mode","answer"), MODE_META["answer"])
    st.markdown(f'<div class="page-header"><div class="ph-eyebrow">Session · {persona.title} · {mode["label"]}</div><h1 class="ph-title">Study with <em>Intention.</em></h1><p class="ph-sub">An offline tutor that thinks with you, not for you.</p><div class="orn-rule"><span class="orn-diamonds">◆ ◆ ◆</span></div></div>', unsafe_allow_html=True)


def render_mode_bar():
    active = st.session_state.get("current_mode","answer")
    chips = "".join(f'<span class="mode-chip {"active" if k==active else ""}">{v["icon"]} {v["label"]}</span>' for k,v in MODE_META.items())
    st.markdown(f'<div class="mode-bar">{chips}</div>', unsafe_allow_html=True)


def render_professor_card_inline():
    key = st.session_state.get("selected_persona","software")
    pmeta = PERSONA_META.get(key, PERSONA_META["software"])
    persona = PERSONAS.get(key, PERSONAS["software"])
    mode = MODE_META.get(st.session_state.get("current_mode","answer"), MODE_META["answer"])
    level = st.session_state.get("last_detected_level","basic")
    st.markdown(f'<div class="inline-prof-card" style="border-color:{pmeta["color"]}33;"><div class="ipc-sigil" style="color:{pmeta["color"]};">{pmeta["sigil"]}</div><div class="ipc-body"><div class="ipc-name" style="color:{pmeta["color"]};">{pmeta["name"]}</div><div class="ipc-dept">{pmeta["dept"]}</div></div><div class="ipc-right"><span class="ipc-mode-icon">{mode["icon"]}</span><span class="ipc-mode-label">{mode["label"]}</span><span class="ipc-level">lvl: {level}</span></div></div>', unsafe_allow_html=True)


def render_chat_message_native(role: str, content: str):
    with st.chat_message("user" if role=="user" else "assistant"):
        st.markdown(content)


def render_rag_sources(sources: list[dict]):
    if not sources: return
    cards = ""
    for i,src in enumerate(sources):
        score = src.get("score")
        score_b = f'<span class="src-score">↑{score:.3f}</span>' if isinstance(score, float) else ""
        fname = src.get("source","Unknown"); page = src.get("page","?")
        snippet = src.get("snippet","")[:240].replace("<","&lt;").replace(">","&gt;")
        cards += f'<div class="src-card"><span class="src-idx">[{i+1}]</span><div class="src-body"><div class="src-title" title="{fname}">{fname}</div><div class="src-meta">p. {page}</div><div class="src-snippet">{snippet}</div></div>{score_b}</div>'
    st.markdown(f'<div class="sources-wrap"><div class="sources-label">Retrieved from your library</div>{cards}</div>', unsafe_allow_html=True)


def render_followups(followups: list[str]):
    if not followups: return
    items = "".join(f'<div class="fup-item"><span class="fup-num">{i+1}.</span>{q}</div>' for i,q in enumerate(followups))
    st.markdown(f'<div class="fups-wrap"><div class="fups-label">Possible next questions…</div>{items}</div>', unsafe_allow_html=True)


def render_quiz(quiz_items: list):
    if not quiz_items:
        st.markdown('<div style="text-align:center;padding:3rem 1rem;"><div style="font-family:var(--serif-d);font-size:2rem;font-style:italic;color:var(--t3);margin-bottom:.75rem;">No Quiz Yet</div><div style="font-family:var(--mono);font-size:.6rem;letter-spacing:.15em;text-transform:uppercase;color:var(--t4);">Click Generate Quiz below the chat</div></div>', unsafe_allow_html=True)
        return
    for idx,item in enumerate(quiz_items,1):
        q_type = "MCQ" if item.question_type=="mcq" else "Short Answer"; opts=""
        if item.question_type=="mcq":
            for oi,opt in enumerate(item.options):
                lbl=chr(65+oi); correct=item.answer_index is not None and oi==item.answer_index
                cls="quiz-opt correct" if correct else "quiz-opt"
                opts+=f'<div class="{cls}"><span class="quiz-opt-lbl">{lbl}</span>{opt}</div>'
            if item.answer_index is not None: opts+=f'<div class="quiz-explanation">✦ {item.explanation}</div>'
        else: opts=f'<div class="quiz-explanation">{item.answer}<br><span style="color:var(--t3);">{item.explanation}</span></div>'
        st.markdown(f'<div class="quiz-card"><div class="quiz-q-num">Q{idx} · {q_type}</div><div class="quiz-q-text">{item.question}</div>{opts}</div>', unsafe_allow_html=True)


def render_ocr_panel(on_image_upload: Callable):
    st.markdown('<div class="sect-divider"><div class="sect-divider-line"></div><span class="sect-divider-text">Handwritten Notes — OCR</span><div class="sect-divider-line"></div></div>', unsafe_allow_html=True)
    img = st.file_uploader("Upload image", type=["jpg","jpeg","png","bmp","tiff"], key="ocr_upload", label_visibility="collapsed")
    if img:
        st.image(img, use_container_width=True)
        if st.button("▸ Extract Text", use_container_width=True): on_image_upload(img.read())
    result = st.session_state.get("ocr_result")
    if result and result.success and result.raw_text:
        st.markdown(f'<div class="ocr-meta-row"><span class="src-score">✓ {result.word_count} words</span><span class="src-score">{result.confidence:.0f}% conf.</span></div>', unsafe_allow_html=True)
        st.text_area("Extracted text", result.raw_text, height=180, label_visibility="collapsed")
        if st.button("📎 Attach to next question", use_container_width=True):
            st.session_state["use_ocr_in_next_query"]=True; st.toast("OCR text queued.", icon="📎")
    elif result and not result.success: st.error(f"OCR failed: {result.error}")


def render_system_health():
    online=st.session_state.get("ollama_online",False); db_chunks=st.session_state.get("db_chunk_count",0)
    model=st.session_state.get("selected_model","n/a"); doc_count=len(st.session_state.get("indexed_docs",[]))
    llm_val="Online" if online else "Offline"; llm_color="var(--ok)" if online else "var(--err)"
    st.markdown(f'<div class="sect-divider" style="margin-bottom:1rem;"><div class="sect-divider-line"></div><span class="sect-divider-text">System Health</span><div class="sect-divider-line"></div></div><div class="health-grid"><div class="health-cell"><div class="health-val" style="color:{llm_color};font-size:1.1rem;">{llm_val}</div><div class="health-lbl">LLM Kernel</div></div><div class="health-cell"><div class="health-val">{db_chunks:,}</div><div class="health-lbl">DB Vectors</div></div><div class="health-cell"><div class="health-val">{doc_count}</div><div class="health-lbl">Documents</div></div><div class="health-cell"><div class="health-val" style="font-size:.75rem;color:var(--t2);">BM25+<br>Vector</div><div class="health-lbl">Retrieval</div></div></div><div class="health-stack"><div class="health-row-item"><span class="health-row-key">model</span><span class="health-row-val">{model}</span></div><div class="health-row-item"><span class="health-row-key">embedding</span><span class="health-row-val">all-MiniLM-L6-v2</span></div><div class="health-row-item"><span class="health-row-key">vector db</span><span class="health-row-val">ChromaDB</span></div><div class="health-row-item"><span class="health-row-key">fusion</span><span class="health-row-val">RRF (k=60)</span></div></div>', unsafe_allow_html=True)
    indexed=st.session_state.get("indexed_docs",[])
    if indexed:
        st.markdown('<div style="margin-top:1rem;"><div class="sources-label">Indexed Documents</div>', unsafe_allow_html=True)
        for doc in indexed:
            src=doc.get("source","?"); pages=doc.get("total_pages","?"); cks=doc.get("total_chunks","?")
            st.markdown(f'<div class="src-card"><span class="src-idx">PDF</span><div class="src-body"><div class="src-title">{src}</div><div class="src-meta">{pages} pages · {cks} chunks</div></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("Setup Guide"):
        st.markdown("**Quick start**\n1. `ollama serve`\n2. `ollama pull mistral:latest`\n3. Upload PDF via sidebar\n4. Ask questions")


def toast_success(m): st.toast(m, icon="✅")
def toast_error(m):   st.toast(m, icon="🚨")
def toast_info(m):    st.toast(m, icon="ℹ️")
def render_header(): render_page_header()
def render_mode_banner(): render_mode_bar()
def render_status_badges(): pass
