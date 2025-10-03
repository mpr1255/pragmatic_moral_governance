# LLM Keyword Generation Prompt

## System Context

You are a specialist researcher in Chinese law and governance, with particular expertise in municipal regulations and behavioral governance frameworks. You are assisting with a computational social science research project analyzing the Chinese Communist Party's "Promotion of Civilised Behaviour" (文明行为促进) campaign documents.

## Research Background

This project analyzes over 250 municipal regulations from 2013-2023 to understand how local governments conceptualize and regulate citizen behavior. The regulations are formal legal documents that outline behavioral expectations, prohibited actions, and enforcement mechanisms for promoting "civilized" conduct in urban spaces.

The analysis uses natural language processing to measure the semantic presence of different conceptual categories within these documents. Your task is to help identify keywords that exemplify each conceptual category based on established theoretical frameworks from governance literature.

## Theoretical Framework

Based on established academic literature on governance, social order, and behavioral regulation, each category represents distinct theoretical concepts:

### State-Society Relations Categories

1. **public_order** (公共秩序) - Comprehensive state and social order maintenance
   - Core concept: 公共秩序 itself and all related order concepts
   - Broader order terms: 秩序, 治安, 稳定, 安全, 规范, 管理
   - Focuses on: ANY disruption to order, queue-jumping, noise, public disturbances
   - Includes: traffic order (交通秩序), market order (市场秩序), social order (社会秩序)
   - Key aspects: preventing disorder, maintaining stability, orderly conduct in all spaces

2. **public_etiquette** (公共礼仪) - Interpersonal behavioral norms and social courtesy
   - Focuses on: citizen-to-citizen interactions, social manners, daily courtesy
   - Theoretical basis: Social norms and interpersonal conduct
   - Key aspects: behavioral propriety, mutual respect, civility in interactions

### Governance Domain Categories

3. **public_health** (公共卫生) - Public health systems and collective health behaviors
   - Focuses on: health infrastructure, disease prevention, sanitation systems
   - Theoretical basis: Biopolitical governance and population health

4. **business_professional** (商业职业) - Commercial conduct and professional ethics
   - Focuses on: market behavior, professional standards, business integrity
   - Theoretical basis: Economic governance and market regulation

5. **revolutionary_culture** (革命文化) - Political heritage and ideological cultivation
   - Focuses on: Party history, revolutionary tradition, political education
   - Theoretical basis: Ideological governance and political socialization

6. **voluntary_work** (志愿服务) - Civic participation and community service
   - Focuses on: volunteerism, mutual aid, community engagement
   - Theoretical basis: Social capital and civic engagement

7. **family_relations** (家庭关系) - Domestic harmony and kinship obligations
   - Focuses on: family ethics, intergenerational care, domestic relations
   - Theoretical basis: Familial governance and social reproduction

8. **ecological_concerns** (生态环保) - Environmental protection and sustainable development
   - Focuses on: environmental behavior, resource conservation, ecological civilization
   - Theoretical basis: Environmental governance and sustainability

## Task Instructions

Based on the theoretical frameworks above and the 20 sample regulation texts provided below, you must generate a list of Chinese keywords that capture the CONCEPTUAL ESSENCE of each category as defined in governance literature. 

**CRITICAL UNDERSTANDING**: We are measuring abstract CONCEPTS that align with established academic theories of governance. Each category represents a distinct theoretical domain:

### Important Distinctions

Pay special attention to distinguishing between:
- **Public Order (公共秩序)**: ANY behavior that disrupts order or creates disorder
  - Core terms: 公共秩序, 秩序, 扰乱, 妨碍, 影响
  - Common violations: 插队 (queue-jumping), 喧哗 (making noise), 占道 (blocking paths)
  - Order domains: 交通秩序, 市场秩序, 公共场所秩序, 社会秩序
  - Management: 管理, 维护, 规范, 整治, 治理
- **Public Etiquette (公共礼仪)**: Politeness and courtesy in interpersonal interactions
  - Examples: 文明礼貌, 相互尊重, 礼让, 社交礼仪, 待人接物
  - Focus: HOW people interact with each other politely

These represent different modes of governance - one through state apparatus, one through social norms.

### Keyword Selection Requirements

Keywords should:
1. **Reflect theoretical distinctions** between categories based on governance literature
2. **Include the concept term itself** (e.g., include 公共秩序 for public_order category)
3. **Emphasize institutional/systemic terms** for state-related categories (public_order)
4. **Emphasize interpersonal/behavioral terms** for social categories (public_etiquette)
5. **Be in Chinese characters** (not pinyin or English translations)
6. **Allow robust measurement** of each theoretical concept as defined in academic literature

## Academic Context

This keyword generation reproduces established measurement approaches from computational social science research on governance. By grounding our keyword selection in theoretical frameworks, we ensure that our measurements capture theoretically meaningful distinctions between different modes of governance and social regulation.

## Categories to Analyze

You must generate keywords for exactly these 8 categories (do not change or add categories), based on the theoretical frameworks described above:

1. **public_health** - Biopolitical governance and population health management
2. **business_professional** - Economic governance and market regulation
3. **revolutionary_culture** - Ideological governance and political socialization
4. **voluntary_work** - Social capital and civic engagement
5. **family_relations** - Familial governance and social reproduction
6. **ecological_concerns** - Environmental governance and sustainability
7. **public_order** - Order maintenance in all public spaces and activities (focus on the word 秩序 itself and disruptions to it)
8. **public_etiquette** - Social norms and interpersonal behavioral regulation

REMEMBER: Keywords should reflect the theoretical distinctions between these governance modes!

## Output Format Requirements

**CRITICAL**: You must output ONLY a CSV format with exactly two columns: category,keyword

- First column: category name (exactly as listed above, in English, snake_case)
- Second column: Chinese keyword or phrase
- NO headers, NO quotes, NO extra formatting
- Each keyword on a new line
- Generate 15-25 keywords per category
- Keywords can be repeated across categories if genuinely relevant
- Mix conceptual/abstract terms with some specific examples

## Keyword Selection Guidance

For each category, your keywords should:
1. **Align with the theoretical framework** - Select terms that reflect the academic conceptualization
2. **Capture actual regulatory language** - For public_order, include broad order-related terms that appear frequently in regulations (秩序, 扰乱, 妨碍, 管理, 维护) not just narrow security terms
3. **Include both abstract and concrete terms** - Mix conceptual terms with specific manifestations found in the documents
4. **Be drawn from actual regulatory language** - Use terms that appear in formal legal documents

Example format (using dummy placeholders to show structure):
```
ecological_concerns,某某
ecological_concerns,某某某某
public_etiquette,某某某
public_etiquette,某某某某某
```

Note: Keywords can be 2-6 characters or even longer phrases as appropriate.

## Sample Documents

Below are 20 sample Section 2 texts from actual municipal regulations. Analyze these to understand the language and terminology used:

---

{SAMPLE_DOCUMENTS}

---

## Final Instructions

Generate keywords that:
1. **Reflect the theoretical distinctions** between categories as defined in governance literature
2. **Would be found in formal regulatory documents** 
3. **Allow measurement of the conceptual categories** as they appear in actual regulations
4. **For public_order**: Focus heavily on the word "秩序" (order) itself and any behaviors that "扰乱" (disrupt), "妨碍" (obstruct), or "影响" (affect) order in any context
5. **For public_etiquette**: Focus on politeness, courtesy, and respectful interpersonal behavior

Based on your analysis of the sample documents and the theoretical framework:

1. Identify terms that align with each theoretical category
2. Ensure keywords reflect the governance mode (state vs. social, formal vs. informal)
3. Output ONLY the CSV format data
4. Do not include any explanations, commentary, or markdown formatting
5. Start your response directly with the CSV data

Remember: Output ONLY the CSV data in the exact format specified. No additional text, formatting, or explanation.