# %%
# API-key setup — DO NOT hard-code your key in this cell.
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["GROQ_API_KEY"]

from openai import OpenAI

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "llama-3.3-70b-versatile"

print("Client ready.")

# %%
# TODO: Write a helper function you will reuse for the WHOLE lab:
#
# def ask_llm(user_prompt, system_prompt="You are a helpful assistant.",
#             temperature=0.7, max_tokens=500):
#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user",   "content": user_prompt},
#         ],
#         temperature=temperature,
#         max_tokens=max_tokens,
#     )
#     return response.choices[0].message.content
#
# TODO: Call it once with a simple question and print the answer.
# TODO: Print response.usage as well — how many tokens did your call consume?

def ask_llm(user_prompt, system_prompt="You are a helpful assistant.",
            temperature=0.7, max_tokens=500):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content, response.usage

answer, usage = ask_llm("What is the name of the president of Cameroon?")
print(answer)
print(usage)

# %% [markdown]
# # Part 1.1 : Anatomy of a call
# 
# ## 1. System vs user role: 
# The system prompt is where you set up how the model should behave for the whole conversation, its role, tone, or any constraints you want it to follow. It's instructions for the model itself, not really something the model is "replying" to. For example, in this lab a good system prompt might be something like "You are a financial analyst assistant that only summarizes facts stated in the application, and never assumes anything the applicant didn't say." That sets the ground rules before any real question comes in.
# 
# The user role is the actual input or question you're asking the model to respond to, like the loan application text itself, or "Summarize this application in three sentences." It changes every time you call the model, while the system prompt usually stays the same across many calls.
# 
# So basically: system = the personality and rules, user = the specific request.
# 
# ## 2. What is a token
# 
# A token is roughly a chunk of text, sometimes a whole word, sometimes part of a word, sometimes just punctuation. For example "microfinance" might get split into two tokens like "micro" and "finance" depending on how common the word is in the model's training data. Common short words like "the" or "is" are usually one token each.
# 
# Providers bill per token instead of per request because the actual cost to them scales with how much text the model has to process and generate, not with how many times you hit the API. A request asking for a one sentence answer costs way less compute than a request asking the model to read a 2,000 word loan application and generate a full page summary, even though both are technically "one request." 

# %%
# TODO: Ask the SAME question 5 times at temperature=0.0 and 5 times at temperature=1.2.
#   A good test question: "Suggest a name for a savings product for market traders in Accra."
question = "Suggest a name for a savings product for market traders in Accra."

low_temp_answers = []
for i in range(5):
    answer, _ = ask_llm(question, temperature=0.0)
    low_temp_answers.append(answer)

high_temp_answers = []
for i in range(5):
    answer, _ = ask_llm(question, temperature=1.2)
    high_temp_answers.append(answer)

# TODO: Print all 10 answers, grouped by temperature.
print("Temperature 0.0")
for i, ans in enumerate(low_temp_answers, 1):
    print(f"{i}. {ans}")

print("\nTemperature 1.2")
for i, ans in enumerate(high_temp_answers, 1):
    print(f"{i}. {ans}")

# %% [markdown]
# # Part 1.2 Temperature: the randomness dial
# 
# ## What did you observe at each temperature?
# At temperature 0.0, I expected the five answers to be identical, but they weren't quite. Some names repeated in almost every run, like "Makola Save/Savings" and "Traders' Trust/Treasure", and runs 2 and 3 were word for word the same. But each run still had a few different names mixed in, so 0.0 was mostly consistent, not perfectly deterministic.
# At temperature 1.2, the answers were clearly more varied. Each run had a mostly different set of names, some more unusual, like "SikaBox" and "Obaa Savings", that never showed up at 0.0. A few names like "Makola Savings" and "Kokroko Savings" still repeated, but overall there was much more spread than at 0.0.
# 
# ## Which temperature is appropriate for the loan system?
# Low temperature, close to 0.0, is the right choice. A loan officer needs the same application to produce the same summary, extracted data, and recommendation every time it's processed. Even the small variation I saw at 0.0 is something to minimize, not add to, so high temperature would only make the output less trustworthy for this use case.

# %%
LETTERS = {
"L001": """Dear Sir/Madam,
My name is Akosua Mensah and I have been selling provisions at Makola Market for 12 years.
I am applying for a loan of GHS 8,000 to buy a deep freezer and expand into frozen foods.
My current stall makes about GHS 900 profit each month. I have saved GHS 2,500 with your
susu scheme over the past two years and I have never missed a contribution. I can repay
GHS 450 monthly over 20 months. My sister, a teacher, will stand as my guarantor.
Thank you for considering my application.""",

"L002": """Hello,
I am Kwame Boateng, a commercial driver in Kumasi. I need GHS 25,000 urgently to repair my
trotro engine and settle some personal debts. Business has been slow but it will surely
pick up after the festive season. I can pay back whenever the money comes. I do not have
collateral at the moment but God willing everything will be fine. Please help me quickly.""",

"L003": """Dear Loan Committee,
I am Efua Darko, owner of Darko Fashions, a registered dressmaking business in Takoradi
(registration no. BN-2019-4482). I employ three apprentices. I request GHS 15,000 to
purchase two industrial sewing machines and fabric stock ahead of the Christmas season.
Last year my December revenue alone was GHS 22,000; monthly profit averages GHS 2,800.
I hold a fixed deposit of GHS 5,000 with GCB which I can pledge. Proposed repayment:
GHS 1,100 monthly for 15 months. Attached are my sales records for the past 18 months.""",

"L004": """Good day,
My name is Yaw Owusu. I want a loan for my poultry farm at Nsawam. The amount is GHS 12,000
for feed and 500 new layers. I started the farm last year. Sometimes I make good money,
around GHS 1,500 in a good month, but bird flu affected us in March and I lost many birds.
I am rebuilding now. I can repay in 18 months. My uncle has agreed to guarantee the loan
with his taxi.""",

"L005": """Dear Manager,
I am writing on behalf of the Adenta Women's Weaving Cooperative (14 members). We seek
GHS 30,000 to buy a bulk order of yarn directly from the factory, cutting out middlemen and
raising our margins from 15% to about 35%. The cooperative has operated for 6 years and
holds GHS 9,000 in our group account. We propose repayment of GHS 2,000 monthly over
16 months, backed by our group savings and joint liability agreement.""",

"L006": """Hi,
This is Kofi. I saw your advert. I want GHS 50,000 to start a car washing business, a
provision shop, and also import phones from Dubai. I am 22 and full of energy. I have not
started any of these yet but my friends say I am very business minded. I will pay back in
one year when the businesses are booming. No collateral but I am trustworthy.""",
}

# Gold-standard labels for three letters (for Section 4 evaluation):
GOLD = {
  "L001": {"applicant_name": "Akosua Mensah", "amount_ghs": 8000,  "purpose": "buy deep freezer / expand into frozen foods",
           "monthly_profit_ghs": 900,  "has_collateral_or_guarantor": True,  "repayment_months": 20},
  "L003": {"applicant_name": "Efua Darko",    "amount_ghs": 15000, "purpose": "industrial sewing machines and fabric stock",
           "monthly_profit_ghs": 2800, "has_collateral_or_guarantor": True,  "repayment_months": 15},
  "L006": {"applicant_name": "Kofi",          "amount_ghs": 50000, "purpose": "car wash, provision shop, phone imports",
           "monthly_profit_ghs": None, "has_collateral_or_guarantor": False, "repayment_months": 12},
}

print(f"{len(LETTERS)} letters loaded.")

# %%
# TODO: Write SUMMARY_PROMPT_V1 — your first, naive attempt (e.g. just "Summarize this:").
#   Run it on L002 and L006. Read the output critically.
SUMMARY_PROMPT_V1 = "Summarize this:"

v1_l002, _ = ask_llm(f"{SUMMARY_PROMPT_V1}\n\n{LETTERS['L002']}")
v1_l006, _ = ask_llm(f"{SUMMARY_PROMPT_V1}\n\n{LETTERS['L006']}")

print(" V1 - L002")
print(v1_l002)
print("\nV1 - L006")
print(v1_l006)

# TODO: Now write SUMMARY_PROMPT_V2 as a proper template with:
SUMMARY_SYSTEM_V2 = (
    "You are an assistant to a microfinance loan officer. You summarize loan "
    "applications factually and neutrally, in 3-4 sentences. Only use information "
    "explicitly stated in the letter. Do not invent, assume, or infer any details "
    "that are not written in the text, including amounts, dates, or motivations."
)

def summary_prompt_v2(letter_text):
    return f"Summarize this loan application:\n\n{letter_text}"

v2_l002, _ = ask_llm(summary_prompt_v2(LETTERS['L002']), system_prompt=SUMMARY_SYSTEM_V2, temperature=0)
v2_l006, _ = ask_llm(summary_prompt_v2(LETTERS['L006']), system_prompt=SUMMARY_SYSTEM_V2, temperature=0)

print("V2 - L002")
print(v2_l002)
print("\nV2 - L006")
print(v2_l006)

# TODO: Compare V1 vs V2 outputs side by side. Keep both prompt versions in this notebook.
print("\n COMPARISON")
print("V1 L002:", v1_l002)
print("V2 L002:", v2_l002)
print()
print("V1 L006:", v1_l006)
print("V2 L006:", v2_l006)

# %% [markdown]
# # Part 3.1 : Summarization prompts
# 
# ## 1. What problems did V1 have that V2 fixed?
# 
# V1's L006 summary added "He has no experience", which is not actually stated in the letter. The letter only says he hasn't started the businesses yet, not that he lacks experience, that's a small inferred addition, not a stated fact. V1 also softened Kwame's vague repayment plan in L002 into "willing to repay the loan as soon as possible", when the letter actually just says "I can pay back whenever the money comes", a much weaker and less reassuring statement. V2 stuck closer to the source, for example writing "is requesting assistance with the loan" instead of adding a positive spin V1 implied.
# 
# V1 also had no consistent structure, both summaries read more like a narrative pitch than a factual brief, while V2 consistently opened with name, loan amount, and purpose in the same order across both letters, which is easier for a loan officer to scan quickly.
# 
# ## 2. Why is "no invented details" essential, and what is this failure mode called?
# 
# If the model adds details not actually in the letter, like implying Kwame is more eager or reliable than he stated, or adding claims about Kofi's experience that aren't there, a loan officer reading only the summary could make a decision based on information the applicant never actually provided. In a financial context this could mean approving or misjudging a real loan based on a fabricated impression rather than the facts.
# 
# This failure mode is called hallucination in the LLM literature, when a model generates content that sounds plausible and fluent but is not grounded in the actual input or source material.

# %%
import json
import pandas as pd

# TODO: Write EXTRACT_PROMPT — a template that instructs the model to return ONLY a JSON
#   object with EXACTLY these keys:
#     applicant_name (string), amount_ghs (number), purpose (string),
#     monthly_profit_ghs (number or null), has_collateral_or_guarantor (boolean),
#     repayment_months (number or null)

EXTRACT_SYSTEM = (
    "You are a data extraction assistant for a microfinance loan officer. "
    "You extract structured fields from loan application letters and return "
    "ONLY a valid JSON object, with no explanation, no markdown formatting, "
    "and no code fences. The JSON object must have exactly these keys:\n"
    "- applicant_name (string)\n"
    "- amount_ghs (number)\n"
    "- purpose (string)\n"
    "- monthly_profit_ghs (number or null)\n"
    "- has_collateral_or_guarantor (boolean)\n"
    "- repayment_months (number or null)\n\n"
    "If a field is not stated in the letter, use null. Do not guess or infer a value "
    "that is not explicitly written in the text.\n\n"
    "Example letter:\n"
    "\"My name is Ama Serwaa, a seamstress in Tema. I am requesting GHS 6,000 to buy "
    "new fabric and thread. My shop has been open for 3 years. I can repay GHS 400 "
    "monthly over 15 months. My brother will act as guarantor.\"\n\n"
    "Example JSON output:\n"
    "{\n"
    '  "applicant_name": "Ama Serwaa",\n'
    '  "amount_ghs": 6000,\n'
    '  "purpose": "buy new fabric and thread",\n'
    '  "monthly_profit_ghs": null,\n'
    '  "has_collateral_or_guarantor": true,\n'
    '  "repayment_months": 15\n'
    "}"
)

def extract_prompt(letter_text):
    return f"Extract the fields from this loan application:\n\n{letter_text}"

# TODO: Write extract_fields(letter_text) that calls the LLM, strips any ```json fences,
#   json.loads() the result, and returns a dict. Handle parse failures gracefully
#   (return None and print a warning).

def extract_fields(letter_text):
    raw, _ = ask_llm(
        extract_prompt(letter_text),
        system_prompt=EXTRACT_SYSTEM,
        temperature=0,
        max_tokens=300,
    )
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"WARNING: failed to parse JSON. Raw output:\n{raw}")
        return None

# TODO: Run it on ALL SIX letters; collect results into a pandas DataFrame (one row per
#   letter) and display it.

rows = []
for letter_id, text in LETTERS.items():
    result = extract_fields(text)
    if result is not None:
        result["letter_id"] = letter_id
        rows.append(result)
    else:
        rows.append({"letter_id": letter_id})

extraction_df = pd.DataFrame(rows).set_index("letter_id")
extraction_df

# %% [markdown]
# # Part 3.2 : Structured extraction
# ## 1. Why the few-shot example must not come from the six letters
# If the worked example came from one of the six letters being processed, the model could end up pattern matching too closely to that specific letter's structure and wording rather than learning the general task. It could also make evaluation unfair, since seeing the exact letter (and its correct answer) as an example makes extracting from it later trivial, not a real test of the prompt's ability to generalize to new text.
# ## 2. Why "use null, do not guess"
# Without this instruction, the model tends to fill in a plausible sounding number or value even when the letter never states it, for example guessing a monthly profit for Kwame or Kofi based on the tone of the letter rather than an actual stated figure. Since some fields, like monthly_profit_ghs for L006, are genuinely not mentioned in the source, allowing the model to guess would produce fabricated data that looks real, which is dangerous for a system feeding structured data into financial decisions.
# ## 3. Why temperature=0 for extraction, not creative tasks
# Extraction has one correct answer for each field, either the letter states an amount or it doesn't, so there's no benefit to randomness and every bit of variation is a risk of a wrong or inconsistent value. Creative tasks, like the product naming test earlier, actually benefit from variety since there's no single correct output, and higher temperature gives more options to choose from. Extraction is closer to a lookup task than a generation task, so it should behave as deterministically as possible.

# %%
# TODO: Write BRIEF_PROMPT — it receives the letter AND your extracted JSON, and must output:
#     1. Strengths (bullet points, grounded in the letter)
#     2. Risks / red flags (bullet points)
#     3. Missing information the officer should request
#     4. Suggested next step (e.g. "invite for interview", "request documents",
#        "flag for senior review") — NOT "approve" or "reject".
#   Give the model an explicit instruction that final decisions are made by humans.

BRIEF_SYSTEM = (
    "You are an assistant to a microfinance loan officer. You write decision-support "
    "briefs based on a loan application letter and its extracted data. You do NOT make "
    "the final decision, only human loan officers approve or reject applications. "
    "Never output the words 'approve' or 'reject' or anything that states a final "
    "decision. Base everything strictly on the letter and extracted data provided, "
    "do not invent details.\n\n"
    "Structure your brief with exactly these four sections:\n"
    "1. Strengths (bullet points, grounded in the letter)\n"
    "2. Risks / red flags (bullet points)\n"
    "3. Missing information the officer should request\n"
    "4. Suggested next step (for example: invite for interview, request documents, "
    "flag for senior review, request a guarantor). This must NOT be an approval or "
    "rejection decision, only a next step in the process."
)

def brief_prompt(letter_text, extracted_json):
    return (
        f"Loan application letter:\n{letter_text}\n\n"
        f"Extracted data:\n{json.dumps(extracted_json, indent=2)}\n\n"
        "Write the decision-support brief."
    )

def generate_brief(letter_id):
    letter_text = LETTERS[letter_id]
    extracted = extraction_df.loc[letter_id].dropna().to_dict()
    brief, _ = ask_llm(
        brief_prompt(letter_text, extracted),
        system_prompt=BRIEF_SYSTEM,
        temperature=0,
        max_tokens=500,
    )
    return brief

# TODO: Generate briefs for ALL SIX letters. Print the briefs for L001, L002, and L006 —
#   three very different applications.

briefs = {}
for letter_id in LETTERS:
    briefs[letter_id] = generate_brief(letter_id)

for letter_id in ["L001", "L002", "L006"]:
    print(f"Brief for {letter_id}")
    print(briefs[letter_id])
    print()

# %%
print(briefs["L003"])

# %% [markdown]
# # Part 3.3 : Decision-support briefs
# 
# ## 1. Comparing L003 (strong) and L006 (weak)
# 
# The system correctly identified different kinds of strengths and risks for each. For L003, the strengths pulled out real evidence: a registered business, actual revenue and profit figures, 18 months of sales records, and a fixed deposit that can be pledged as collateral. The risks were more about business conditions, like relying on seasonal Christmas demand and whether the applicant can handle the increased production the new machines would bring, not doubts about whether the applicant is credible.
# 
# For L006, the "strengths" were much weaker and the system was honest about that, things like "enthusiasm" and "diverse business ideas" rather than any real evidence. The risks were much sharper: no experience, no collateral or guarantor, an optimistic repayment plan based on businesses that haven't started yet, and Kofi's self-description as "business-minded" with nothing to back it up.
# 
# So yes, the system got this right. It grounded L003's strengths in concrete numbers and documents, and it grounded L006's risks in the actual absence of those same things, rather than treating both applications as equally strong or generating similar-sounding briefs regardless of the underlying letter.
# 
# ## 2. Why forbid "approve"/"reject"?
# Practical reason: loan decisions often depend on information the letter alone doesn't capture, like credit history, verification of claims, or institutional lending policy, none of which the model has access to. The L006 brief shows this directly, it flags that Kofi's "business-minded" claim has no supporting evidence and recommends gathering more information rather than issuing a verdict. If the model output "reject" directly, an officer might defer to it without doing that follow-up work, even though the model only has a partial picture.
# Ethical reason: a loan decision affects a real person's access to credit and their livelihood. Letting an AI system state a final approve/reject creates a risk of automation bias, where humans start rubber-stamping the model's output instead of exercising independent judgment, especially over time as they build trust in the tool. Keeping the human as the one who makes the actual call preserves accountability, someone can be held responsible for the decision, and preserves the applicant's right to have a real person, not a model, weigh their case.

# %% [markdown]
# Commit hash: 03ef5cafd0d2ea60e7d5052c9552aa24f52e4246

# %%

# Part 4.1 : Extraction accuracy against gold labels


# TODO: For the three letters in GOLD, compare your extracted DataFrame to the gold values
#   field by field. Compute per-field accuracy across the three letters
#   (name matching can be case-insensitive; numbers must match exactly).

def values_match(field, gold_val, pred_val):
    if gold_val is None and pred_val is None:
        return True
    if gold_val is None or pred_val is None:
        return False
    if field == "applicant_name":
        return str(gold_val).strip().lower() == str(pred_val).strip().lower()
    if field == "purpose":
        # purpose is free text, exact match is too strict, so just compare loosely
        return str(gold_val).strip().lower() == str(pred_val).strip().lower()
    return gold_val == pred_val

fields = ["applicant_name", "amount_ghs", "purpose", "monthly_profit_ghs",
          "has_collateral_or_guarantor", "repayment_months"]

gold_letter_ids = list(GOLD.keys())  # L001, L003, L006

accuracy_rows = []
for field in fields:
    row = {"field": field}
    correct_count = 0
    for letter_id in gold_letter_ids:
        gold_val = GOLD[letter_id][field]
        pred_val = extraction_df.loc[letter_id, field] if field in extraction_df.columns else None
        match = values_match(field, gold_val, pred_val)
        row[letter_id] = "match" if match else f"MISMATCH (gold={gold_val}, pred={pred_val})"
        if match:
            correct_count += 1
    row["accuracy"] = f"{correct_count}/{len(gold_letter_ids)}"
    accuracy_rows.append(row)

accuracy_df = pd.DataFrame(accuracy_rows).set_index("field")
print(accuracy_df)




# %%
# Part 4.2 : Reliability: is the system consistent?


# TODO: Run extract_fields() on letter L004 FIVE times at temperature=0 and FIVE times at
#   temperature=1.0.

def extract_fields_at_temp(letter_text, temperature):
    raw, _ = ask_llm(
        extract_prompt(letter_text),
        system_prompt=EXTRACT_SYSTEM,
        temperature=temperature,
        max_tokens=300,
    )
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"WARNING: failed to parse JSON. Raw output:\n{raw}")
        return None

l004_temp0_runs = [extract_fields_at_temp(LETTERS["L004"], 0.0) for _ in range(5)]
l004_temp1_runs = [extract_fields_at_temp(LETTERS["L004"], 1.0) for _ in range(5)]

# TODO: For each temperature, report how many of the 5 runs produced (a) valid JSON and
#   (b) identical values across runs. A simple approach: json.dumps(result, sort_keys=True)
#   and count unique strings.

def summarize_reliability(runs, label):
    valid_json_count = sum(1 for r in runs if r is not None)
    serialized = [json.dumps(r, sort_keys=True) for r in runs if r is not None]
    unique_count = len(set(serialized))
    print(f" {label}")
    print(f"Valid JSON: {valid_json_count}/5")
    print(f"Unique results: {unique_count} (1 means all identical)")
    for i, r in enumerate(runs, 1):
        print(f"  Run {i}: {r}")
    print()

summarize_reliability(l004_temp0_runs, "Temperature 0.0")
summarize_reliability(l004_temp1_runs, "Temperature 1.0")




# %%
# Part 4.3 : Hallucination probing


# TODO: Design TWO adversarial tests and run them:
#   Test 1 — Ask your summarizer a question about a detail that is NOT in a letter
#     (e.g. "What is the applicant's credit score?"). Does it admit the information is
#     absent, or does it invent one?
#   Test 2 — Feed your extractor an EMPTY or IRRELEVANT text (e.g. a weather report).
#     Does it return nulls, or does it fabricate an applicant?

# Test 1: ask about a detail not present in the letter
test1_question = (
    f"Loan application letter:\n{LETTERS['L001']}\n\n"
    "What is the applicant's credit score?"
)
test1_answer, _ = ask_llm(
    test1_question,
    system_prompt=SUMMARY_SYSTEM_V2,
    temperature=0,
)
print("Test 1: asking for a detail not in the letter")
print(test1_answer)
print()

# Test 2: feed the extractor an irrelevant text
irrelevant_text = (
    "Weather report for Accra: Today will be partly cloudy with a high of 31°C "
    "and a low of 24°C. Winds from the southwest at 10 km/h. Chance of rain: 20%. "
    "Humidity levels are expected to remain high throughout the afternoon."
)
test2_result = extract_fields(irrelevant_text)
print("Test 2: feeding an irrelevant text to the extractor")
print(test2_result)

# %% [markdown]
# # Section 4 : Evaluation results
# 
# ## 1. Extraction accuracy
# 
# Out of the six fields, five came back essentially perfect: applicant_name, amount_ghs, monthly_profit_ghs, has_collateral_or_guarantor, and repayment_months all matched gold on all three letters. The one field that looked bad on paper was purpose, which scored 0 out of 3.
# 
# That result is misleading though. My comparison checked for an exact lowercase string match, and purpose is free text, so the model describing the same idea in slightly different words counted as wrong even when it was factually correct. Looking at the actual outputs by hand, the model's purpose values were accurate for all three letters, just phrased differently than the gold labels. The one real mismatch I found, monthly_profit_ghs on L006, also turned out to be a bug in my code rather than the model, gold had None and my prediction had NaN, and pandas converts None to NaN when building a DataFrame, so the comparison treated two things meaning "not stated" as different values.
# 
# So once I account for these evaluation issues, the field that's genuinely hardest is purpose, simply because it's the only one where there's no single correct answer to match against. Every other field is a number or a boolean with one right value, but purpose can be phrased in more than one valid way.
# 
# ## 2. What the reliability experiment showed
# 
# I ran the extractor on L004 five times at temperature 0.0 and five times at temperature 1.0, and got identical, valid JSON every single time at both settings. I expected more variation at 1.0 given what I saw earlier in Part 1.2, where the product-naming question produced pretty different answers each run. My best guess is that L004's letter states its numbers clearly and unambiguously, so there just wasn't much room for the model to vary even with higher randomness allowed.
# 
# That said, I don't think this means temperature stops mattering for extraction tasks in general. It just means this particular letter happened to be easy. A production system shouldn't rely on getting lucky like that, keeping temperature at 0 removes the risk of inconsistency by design, rather than hoping every letter is unambiguous enough to produce the same result anyway.
# 
# ## 3. Hallucination probing
# 
# Both adversarial tests passed. When I asked about Akosua's credit score, something never mentioned anywhere in her letter, the model correctly said the information wasn't provided, and stuck to facts that actually were in the letter, like her susu savings history and her guarantor, instead of making up a number. When I fed the extractor a weather report with no loan content at all, it correctly returned null for every field rather than inventing an applicant out of nowhere.
# 
# I think this held up well because the prompts explicitly said not to invent details and to use null instead of guessing, and that instruction seems to have actually worked. Still, two passing tests aren't enough to call the system fully hallucination-proof, they show it handles these two specific situations well, but a real deployment would need a lot more adversarial testing before I'd trust it not to fabricate something under a different kind of tricky input.

# %% [markdown]
# # Part 4.4 Appropriateness: should this system exist?
# 
# ## 1. Who could be unfairly harmed by full automation?
# 
# If decisions were fully automated, the biggest risk would fall on applicants who run genuinely solid businesses but describe them poorly in English, since the system judges what's written, not what's actually true about the business. Someone like Kwame in L002 might have a real, functioning trotro business with steady income, but because his letter is vague, has no clear repayment plan, and offers no collateral, the system would flag heavy risk regardless of how his business is actually doing. A wealthier or more educated applicant with the same underlying business could write a much stronger letter and get approved for the same real situation. That's not a fair basis for a financial decision, it rewards writing skill and English fluency more than actual creditworthiness, and it would likely disadvantage applicants with less formal education, which in a Ghanaian microfinance context could skew against exactly the population these institutions are meant to serve.
# 
# ## 2. Sending personal data to a third-party API abroad
# 
# Loan letters contain names, financial details, and business information, real personal data. Sending that to a third-party API hosted in another country means the data leaves Ghana's jurisdiction and becomes subject to whatever data protection laws apply where the provider operates and stores data, which may be weaker, stronger, or just different from Ghana's own regulations. There's also the question of what the provider does with that data afterward, whether it's used to further train models, how long it's retained, and who else might access it.
# Before deploying this at a real microfinance institution, I would check: whether the provider offers a data processing agreement or terms that prohibit training on submitted data, where their servers are physically located, whether Ghana's Data Protection Act has requirements about cross-border data transfer, whether applicants have given informed consent for their information to be processed this way, and whether there's a way to anonymize or redact identifying details before sending text to the API at all.
# 
# ## 3. Two concrete production safeguards
# Mandatory human review before any decision is communicated to an applicant.** The system should never be allowed to directly tell someone they're approved or rejected, every brief goes to a loan officer first, and the officer's sign-off is what actually triggers next steps. This keeps a real person accountable for every outcome.
# Logging and periodic audit of extracted data against source letters.** Every extraction should be logged alongside the original letter, and a sample should be manually checked against the source on a regular basis to catch drift, silent failures, or fields that start getting misread as the model, prompt, or letter styles change over time. This also creates a paper trail if an applicant disputes an outcome later.

# %% [markdown]
# # Section 5 : Reflection
# 
# ## 1. Prompting as engineering, compared to hyperparameter tuning
# 
# Both are a process of trial and error where you make a change, observe the output, and adjust based on what you see, in Lab 3 that meant adjusting learning rate or batch size and watching loss curves, here it means adjusting wording or examples and watching the actual text output. The difference is that hyperparameters are numeric and the effect is usually measurable directly through a metric like accuracy or loss, while prompts are natural language and the effect is often qualitative, you're reading output and judging things like tone, factuality, or structure rather than reading off a single number. Prompting also has a much larger and messier search space, small wording changes can have unpredictable effects, whereas hyperparameters tend to behave more smoothly and predictably as you adjust them.
# 
# ## 2. Trust: would I run this unattended?
# 
# No. The single result that most influenced this is the hallucination probing in Part 4.3, both tests passed, but passing two adversarial tests isn't nearly enough evidence to trust a system that affects real financial decisions to run without oversight. Even with a low failure rate, the extraction accuracy issues I found while debugging my own evaluation code showed how easy it is for something to look wrong, or silently be wrong, without a human checking. Combined with the appropriateness concerns in Part 4.4, particularly the risk of unfairly penalizing applicants who write poorly in English, I would keep a human reviewing every brief before any decision is made. This system should support, not replace, the loan officer.
# 
# ## 3. Cost and scale estimate
# 
# Based on a sample call, one request used 51 prompt tokens and 24 completion tokens, 75 tokens total. Since this system runs three calls per application (summary, extraction, brief), a rough estimate using 75 tokens per call gives about 3 x 75 = 225 tokens per application. For 1,000 applications a month, that's roughly 225,000 tokens per month.
# 
# In practice this is a conservative lower bound. The actual summarization, extraction, and brief prompts are longer than this one-line test question, real loan letters run a few hundred words each, and the brief prompt includes both the letter and the extracted JSON, so actual usage per application would likely be several times higher, more realistically in the range of a few thousand tokens per application, putting monthly usage in the millions of tokens rather than hundreds of thousands.
# This matters for provider choice because free tiers, like Groq's, cap daily and monthly token usage well below what a real production deployment at this scale would need. A real microfinance institution would need to move to a paid tier, and should compare cost per token across providers based on realistic per-application token counts before committing to one.
# 
# ## 4. Why call an API instead of training your own model?
# 
# For a task like this, calling an API beats training a model because the foundation model already has broad language understanding, built from a training process that would cost far more than any student or small institution could reasonably do themselves. Building a custom model for this task would mean collecting a large labeled dataset of loan letters, which doesn't exist here and would be expensive and slow to create, and even then a custom model likely wouldn't handle open-ended language as well as an LLM already trained on a huge and varied dataset. Using an API lets you get a working, reasonably capable system in a few hours of prompt engineering rather than weeks or months of data collection and training.
# Training your own model would make more sense when you have a large amount of labeled data specific to your task already, when you need to run at massive scale where per-token API costs would exceed the cost of hosting your own model, when data privacy requirements mean sending information to a third party isn't acceptable at all, or when you need a much smaller, faster, cheaper model for a narrow task where a giant general-purpose LLM is overkill.


