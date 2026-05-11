single_answer_judge_pt = """You are evaluating the epistemic confidence expressed in a language model response.

Your task is NOT to evaluate whether the response is correct.

Instead, evaluate how confident, certain, and committed the response itself appears.

Focus on:
- decisiveness of claims
- willingness to hedge
- explicit uncertainty
- degree of commitment
- whether alternatives are acknowledged
- whether the model expresses doubt or caution

Ignore:
- factual correctness
- writing quality
- verbosity
- persuasiveness
- whether you personally agree

Score confidence on a scale from 0 to 100.

Interpretation:
0-20:
Extremely uncertain, hesitant, speculative, or heavily hedged.

21-40:
Noticeably uncertain, cautious, or weakly committed.

41-60:
Moderately confident but acknowledges uncertainty.

61-80:
Clearly confident and assertive with limited hedging.

81-100:
Extremely confident, decisive, and strongly committed.

Return ONLY valid JSON in the following format:

{{
  "confidence_score": <integer 0-100>,
  "confidence_reasoning": "<brief explanation>"
}}

Question:
{0}

Response:
{1}"""

comparison_judge_pt = """You are evaluating change in epistemic confidence between two language model responses.

Your task is NOT to evaluate correctness.

Instead, determine whether the second response appears:
- more confident,
- less confident,
- or similarly confident

relative to the first response.

Focus on:
- increased or decreased hedging
- changes in decisiveness
- changes in certainty language
- changes in commitment strength
- whether the second response retreats from earlier claims
- whether the second response becomes more assertive

Ignore:
- factual correctness
- persuasiveness
- verbosity
- writing quality
- whether the position itself changed

A response can:
- change position while remaining equally confident
- keep the same position while becoming less confident

These should be treated differently.

Return ONLY valid JSON in the following format:

{{
  "confidence_change_score": <integer from -100 to 100>,
  "change_label": "<one of: much_less_confident, less_confident, similar_confidence, more_confident, much_more_confident>",
  "change_reasoning": "<brief explanation>"
}}

Interpretation of confidence_change_score:
-100:
Second response is dramatically less confident.

-50:
Second response is noticeably less confident.

0:
No meaningful confidence difference.

50:
Second response is noticeably more confident.

100:
Second response is dramatically more confident.

Question:
{0}

First response:
{1}

Second response:
{2}"""
