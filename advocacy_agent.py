from dotenv import load_dotenv
import os

load_dotenv()

def generate_report():
    name = "HERizon User"
    issue = "Women are undervalued in invisible labor"

    report = """
HERizon Sovereign - Advocacy Report

User: """ + name + """

Issue:
""" + issue + """

Summary:
Women often perform invisible labor that goes unrecognized.
This AI advocacy agent highlights these contributions
and creates awareness for fair recognition.

Recommendation:
- Recognize unpaid work
- Promote equality
- Encourage policy change

Conclusion:
HERizon Sovereign empowers women by making invisible work visible.
""".strip()

    return report


def save_report(report):
    with open("advocacy_report.txt", "w") as f:
        f.write(report)

    print("\n✅ Report saved to advocacy_report.txt")


if __name__ == "__main__":
    report = generate_report()
    save_report(report)