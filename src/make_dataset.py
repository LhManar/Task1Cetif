"""Génère un dataset synthétique de tickets de support (aucune donnée personnelle)."""
import random
import pandas as pd

random.seed(42)

TEMPLATES = {
    "billing": [
        "I was charged twice for my {plan} subscription this month.",
        "Can I get a refund for the {plan} plan? I cancelled yesterday.",
        "My invoice shows the wrong amount, it should be {amt} dollars.",
        "Where can I download the invoice for last month's payment?",
        "The payment failed but my card was still debited.",
        "Please update my billing address on the invoice.",
        "I see an unknown charge of {amt} dollars on my statement.",
        "How do I change my payment method to a new credit card?",
        "I want to cancel auto-renewal so I am not billed again.",
        "The discount code was not applied to my {plan} order.",
    ],
    "technical": [
        "The app crashes every time I upload a photo.",
        "I get error {code} when I try to open the dashboard.",
        "The page keeps loading forever and never shows my data.",
        "Notifications stopped working after the latest update.",
        "The export button does nothing when I click on it.",
        "Sync between my phone and laptop is broken.",
        "The website is very slow and times out with error {code}.",
        "Search returns no results even for items that exist.",
        "The mobile app freezes on the {plan} screen.",
        "Videos will not play, I only see a black screen.",
    ],
    "account": [
        "I cannot reset my password, the email never arrives.",
        "I forgot my username and cannot log in.",
        "Please change the email address linked to my account.",
        "My account was locked after several login attempts.",
        "How do I enable two-factor authentication?",
        "I want to delete my account and all my data.",
        "I cannot update my profile picture or my display name.",
        "Someone else may have accessed my account, please help.",
        "Can I merge my two accounts into a single one?",
        "The verification code for login is not working.",
    ],
    "general": [
        "Do you offer a student discount?",
        "What are your customer support opening hours?",
        "Is there a way to suggest a new feature?",
        "Do you have an office in {city}?",
        "Great service, I just wanted to say thank you!",
        "Can you tell me more about the {plan} plan features?",
        "Is your product available in other languages?",
        "How can I become a partner or reseller?",
        "Where can I find your terms of service?",
        "Do you plan to release a desktop version soon?",
    ],
}

FILL = {
    "plan": ["Basic", "Pro", "Premium", "Team", "Starter"],
    "amt": ["9", "19", "29", "49", "99"],
    "code": ["404", "500", "503", "E102", "X77"],
    "city": ["Tunis", "Paris", "Berlin", "Madrid", "London"],
}
PREFIX = ["", "", "", "Hello, ", "Hi team, ", "Please help: ", "Urgent! ", "Good morning, "]
SUFFIX = ["", "", "", " Thanks.", " Thank you in advance.", " Please reply asap.", " Regards."]


def typo(s: str) -> str:
    if len(s) < 8 or random.random() > 0.15:
        return s
    i = random.randrange(1, len(s) - 2)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]  # swap two letters


def make_text(label: str) -> str:
    t = random.choice(TEMPLATES[label])
    for k, v in FILL.items():
        t = t.replace("{" + k + "}", random.choice(v))
    t = random.choice(PREFIX) + t + random.choice(SUFFIX)
    if random.random() < 0.15:
        t = t.lower()
    return typo(t)


rows = []
for label in TEMPLATES:
    seen = set()
    while len(seen) < 70:
        seen.add(make_text(label))
    rows += [(t, label) for t in seen]

# Tickets ambigus (volontairement difficiles), étiquetés selon le sujet principal
rows += [
    ("I was charged and now I cannot log in to my account.", "billing"),
    ("Can't log in and the app crashes on the login screen.", "technical"),
    ("Is it possible to get a refund? Also do you have a mobile app?", "billing"),
    ("help", "general"),
    ("My password reset page shows error 500.", "technical"),
    ("Thanks, but I still cannot change my email.", "account"),
    ("Invoice page is not loading.", "technical"),
    ("Do you accept payment by bank transfer?", "billing"),
    ("Why is my account showing the Free plan after I paid?", "billing"),
    ("Is there an API? I could not find it in my account settings.", "general"),
]

df = pd.DataFrame(rows, columns=["text", "label"]).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/tickets.csv", index=False)
print(df.shape)
print(df["label"].value_counts())
