"""
Dorm-Net: Interactive Chat Interface
Run this file to start a terminal chat session with your AASTU AI tutor.
"""
from brain_module import ask_aastu_senior, add_golden_summary

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        🎓 DORM-NET — AASTU Offline AI Peer Tutor        ║
║           Powered by Llama 3.2 + ChromaDB               ║
║                  100% Offline Ready ✅                   ║
╚══════════════════════════════════════════════════════════╝

Commands:
  /add   → Add new course notes to memory
  /help  → Show this help message
  /quit  → Exit the tutor
  
Just type your question to ask the AASTU Senior Tutor!
"""

def add_note_interactively():
    """Guides the user to add a new course note."""
    print("\n📝 ADD NEW COURSE NOTE TO MEMORY")
    print("─" * 40)
    doc_id   = input("Document ID (e.g. 'physics_ch3'): ").strip()
    course   = input("Course name (e.g. 'Physics II'):   ").strip()
    print("Paste your note text below. Type END on a new line when done:")
    
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    
    text = "\n".join(lines)
    if text.strip():
        add_golden_summary(text, document_id=doc_id, metadata={"course": course})
    else:
        print("⚠️  No text entered. Skipping.")

def main():
    print(BANNER)
    
    while True:
        try:
            user_input = input("\n🧑‍🎓 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye! Good luck with your studies!")
            break

        if not user_input:
            continue
        
        if user_input.lower() == "/quit":
            print("👋 Goodbye! Good luck with your studies!")
            break
        elif user_input.lower() == "/help":
            print(BANNER)
        elif user_input.lower() == "/add":
            add_note_interactively()
        else:
            print()
            answer = ask_aastu_senior(user_input)
            print(f"\n🤖 AASTU Senior: {answer}")
            print("\n" + "─" * 60)

if __name__ == "__main__":
    main()
