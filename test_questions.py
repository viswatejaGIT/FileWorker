
test_cases = [
    {
        # Tests: exact year + tricky (document has contradicting info - "soorty it was year 2020")
        "question": "When did the Government of India's holding in IRCTC reduce to 67%?",
        "ground_truth": "In December 2021 (the document notes it was actually year 2020), the Government divested 20%, reducing holding to 67%."
    },
    {
        # Tests: what is RDS - tests less info questions
        "question": "What is RDS?",
        "ground_truth": "RDS stands for Rolling Deposit Scheme, a type of closed wallet introduced in 2012 where customers can book tickets using money already deposited."
    },
    {
        # Tests: similar numbers confusion (67% vs 62.4% vs 87%)
        "question": "What is the current government ownership percentage of IRCTC?",
        "ground_truth": "62.4%, after the government disinvested 5% in December 2022, reducing from 67% to 62.4%."
    },
    {
        # Tests: iMudra vs RDS confusion (two different payment products)
        "question": "What is iMudra and when was it launched?",
        "ground_truth": "iMudra is a payment wallet launched by IRCTC on 7 August 2019. It is different from RDS which is a closed wallet scheme."
    },
    {
        # Tests: Navaratna status - recent fact
        "question": "What is IRCTC's current public sector status?",
        "ground_truth": "As of March 2025, IRCTC was accorded Navaratna status. Previously in May 2008 it was classified as a Miniratna public corporation."
    },
    {
        # Tests: hallucination - document mentions IRCTC revenue/financials NOWHERE
        # but talks about ticketing, services, etc. LLM might hallucinate revenue numbers
        "question": "What is IRCTC's annual revenue?",
        "ground_truth": "I don't have enough information. The document does not mention IRCTC's annual revenue."
    }
]
