# Документація промптів

Усі промпти зібрано у файлі [`ai_analyst/prompts.py`](ai_analyst/prompts.py). Тут наведено їхній текст, пояснення структури та історію ітерацій.

## Загальні принципи

1. **Модель не рахує, а інтерпретує.** Будь-яке число, яке бачить користувач, спочатку обчислює pandas і передає в промпт у блоці `<facts>` як JSON. LLM лише пояснює ці числа. Причина в тому, що LLM упевнено вигадує числа, тоді як pandas рахує точно і відтворювано.
2. **XML-теги для структури** (`<facts>`, `<rules>`, `<task>`, `<environment>`), як рекомендує Anthropic Prompt Engineering Guide. Теги чітко відокремлюють дані від інструкцій, а моделі простіше посилатися на «дані з `<facts>`».
3. **Структурований вихід замість вільного тексту.** Кожна функція має Pydantic-схему (`EdaInterpretation`, `HypothesisSet`, `PandasAnswer`, `AnomalyExplanation`, `ExecutiveSummary`), яку API гарантує через structured outputs (`messages.parse` у SDK або `--json-schema` у CLI). Парсити поля JSON значно надійніше, ніж витягувати їх із тексту.
4. **Інструкції англійською, відповіді українською.** Інструкції англійською модель виконує найстабільніше. Мову відповіді задає окреме правило, а назви колонок і значення лишаються як у даних.
5. **Перевірка після моделі.** Навіть із правилом «бери числа лише з `<facts>`» відповідь проходить автоматичну перевірку `verify_numbers`. Правило в промпті зменшує кількість помилок, а перевірка в коді їх ловить.

## Спільні правила (`COMMON_RULES`)

Додаються в кінець промптів EDA, гіпотез, аномалій і summary:

```
<rules>
- All numbers you mention must be copied from the data inside <facts>. Never compute new numbers, never round
  differently, never invent values that are not there. If a needed number is missing, say what is missing instead.
- Write every text field in Ukrainian. Keep column names, segment values and code exactly as in the data.
- Be specific: name the column, the segment and the number. Avoid generic advice that would fit any dataset.
</rules>
```

Чому так. Перше правило – головний захист від галюцинацій. Фраза «never round differently» з'явилася тому, що округлення (наприклад, 28 636,64 → «майже 29 тисяч») ускладнює автоматичну перевірку. Пункт «if a needed number is missing, say what is missing» дає моделі легальний вихід замість вигадування. Це видно в демо: у summary модель сама написала, що бракує даних про витрати на залучення та конверсію. Третє правило бореться з загальними порадами на кшталт «варто покращити маркетинг».

## 1. Автоматичний EDA (`EDA_SYSTEM`)

```
You are a senior product analyst reviewing a new dataset before analysis.
You receive a profile of a CSV file computed by pandas. Interpret it for another analyst: what the data is,
which data quality problems must be fixed before analysis, what the distributions of key variables say,
and the first conclusions.

<what_to_look_for>
- missing values, duplicate rows, inconsistent spellings of the same category (case, spaces, synonyms),
  mixed date formats (see raw_value_formats), dates far after the last regular day (future dates),
  negative or zero amounts, extreme outliers, identifier columns that should not be aggregated;
- for each problem give the column, the evidence with numbers from the profile, severity and a concrete fix.
</what_to_look_for>
+ COMMON_RULES
```

Вхід – профіль від `profiling.profile_dataframe`. Для кожної колонки він містить тип, пропуски, кількість унікальних значень, квантилі, викиди за правилом 3×IQR, групи різних написань (після `strip().lower()`), «форму» значень (`9999-99-99` проти `99.99.9999`) і останній регулярний день у колонці дат.

Чому такий список у `<what_to_look_for>`. Пропуски й розподіли видно одразу, а мішані формати дат і «дати з майбутнього» не помітні без спеціальних полів профілю. Тому кожному пункту списку відповідає окреме поле профілю, і промпт прямо називає його (`see raw_value_formats`). Окремий пункт про колонки-ідентифікатори потрібен тому, що `user_id` у цьому датасеті читається як число (float64 через пропуски) і інакше може потрапити в «середні значення».

Схема `QualityIssue` має поля `evidence` (числа з профілю) і `recommendation` (конкретна дія). Вони змушують модель доводити кожну проблему цифрами. Результат: 9 проблем із доказами, перевірка 150 з 150 чисел.

## 2. Генерація гіпотез (`HYPOTHESES_SYSTEM`)

```
You are a senior product analyst. You receive a short dataset description and a numbered
list of candidate observations (<facts>, each with an id like F7) computed by pandas. Choose the observations that
are most worth investigating and turn them into 5-10 testable hypotheses for the product team.

<hypothesis_requirements>
- Each hypothesis is grounded in at least one fact: list the fact ids and quote their numbers in "evidence".
- Prefer large, actionable differences between segments, trends and mix shifts over trivial ones.
- Do not present the same observation twice. Skip facts that are obvious artefacts of identifier columns.
- "how_to_check" names a concrete analysis (which columns, which comparison or test).
- priority: high = big effect on a large segment, low = small or uncertain.
</hypothesis_requirements>
+ COMMON_RULES
```

### Історія ітерацій

**v0 – наївний промпт** (збережено в `experiments/prompt_v0_hypotheses.py`, результат – `examples/06_prompt_v0_hypotheses.json`):

```
You are a data analyst. Look at the data and suggest 5-10 hypotheses worth checking. Answer in Ukrainian.
```

На вхід подавалися `df.describe()` і `df.head(10)`. Модель дала 10 гіпотез, але частина з них спиралася на 10 рядків із `head()`: «користувач 20049 заплатив 7.49 і через 22 с – 17.99», «нестандартні суми 8.72, 10.44». Це анекдоти, а не закономірності. Деякі твердження перевірити було нічим, наприклад «Україна має нижчий середній чек, ніж країни Заходу»: у `describe()` такого розрізу немає. Перевірка чисел: **67 з 82** підтверджено, 15 значень у даних не знайдено.

**v1 – факти замість сирих даних.** Розрахунок кандидатних спостережень перенесено в pandas (`hypothesis_facts`): середні значення метрик у сегментах, частки результату (outcome rate) у сегментах, тижневі тренди, пікові тижні, дні тижня, зсув структури між половинами періоду. Кожен факт отримав id (F1…F40), а модель має посилатися на ці id. Проблема v1, виявлена на `freemium_events.csv`: верх рейтингу займали структурні залежності між колонками, наприклад «серед `event_params = plan=premium` частка `event_name = start_trial` = 100 %». Це не інсайт, а спосіб запису даних. Крім того, «iOS», « iOS» та «ios» рахувалися як різні сегменти.

**v2 – поточна версія.** Факти рахуються на нормалізованих категоріях. Частки 0 % і 100 % відкидаються як структурні. Лишаються лише статистично значущі відхилення: |z| ≥ 3 для часток і для середніх. Так прибрано «ефекти» на 3–4 рядках, наприклад частку 0,14 % проти 0,42 %. У промпт додано вимоги не повторювати спостереження і пропускати артефакти ідентифікаторів. Після генерації код видаляє з відповіді id фактів, яких не існує. Результат на тих самих даних: 9 гіпотез, кожна з посиланням на факти, перевірка **89 з 89** чисел.

## 3. Natural language → pandas (`NL2PANDAS_SYSTEM`)

```
You translate an analyst's question into pandas code over a DataFrame named `df`.

<environment>
- Available names: df (pandas DataFrame, already loaded), pd, np. No imports, no file or network access,
  no attributes that start with an underscore, no df.query/eval.
- The code must assign the final answer to a variable named `result`
  (a DataFrame, Series or scalar; for "top N" questions return a sorted DataFrame with readable column names).
- Datetime columns are already parsed. Category values may have inconsistent case or spaces - normalise with
  .str.strip().str.lower() when filtering on text values.
</environment>

<when_you_cannot_answer>
If the question cannot be answered from the columns of this dataset (a column does not exist, the question is not
about the data, or it is ambiguous beyond a reasonable assumption), set can_answer to false, explain why in
reason_if_not (in Ukrainian) and leave code empty. Never invent columns.
</when_you_cannot_answer>

Write "explanation" in Ukrainian: 1-3 sentences on what the code computes and which assumptions it makes.
```

Вхід – схема датасету (`schema_for_llm`): колонки, типи, кількість пропусків, допустимі значення для категорій до 30 унікальних, діапазон дат і 5 прикладів рядків. Повний датасет у промпт не передається: це дорого і не потрібно, бо код виконується локально на всіх даних.

Чому так:
- Блок `<environment>` дослівно описує обмеження пісочниці (`safe_exec.py`). Якщо модель знає правила наперед, вона рідше пише код, який пісочниця відхилить.
- Вимога записати відповідь у `result` дає однозначний контракт між моделлю та виконавцем.
- Підказка про нормалізацію категорій потрібна для «брудних» даних: фільтр `df['platform'] == 'iOS'` мовчки пропускає « iOS» та «ios».
- Блок `<when_you_cannot_answer>` разом із полем `can_answer` у схемі – це обробка нелогічних запитів. На питання «Який прогноз погоди в Києві?» інструмент відповідає, що таких даних немає, а не вигадує код (`examples/03c_ask.md`).
- Поле `explanation` з припущеннями робить логіку прозорою. На питання про retention без даних про активність модель пояснила, що вважає активністю платіж (`examples/03d_ask.md`).

**Автовиправлення (`NL2PANDAS_FIX`).** Якщо код падає або не проходить перевірку пісочниці, текст помилки повертається моделі разом із попереднім кодом:

```
The previous code failed. Fix it and return the full corrected code.
<previous_code>…</previous_code>
<error>…</error>
```

Дозволено одну спробу виправлення. Це покриває типові помилки (неправильна назва колонки, тип даних) і не дає зациклитися. Сценарій «помилка → виправлення» перевіряє офлайн-тест `test_ask_repairs_failing_code`.

## 4. Пояснення аномалій (`ANOMALY_SYSTEM`)

```
You are a senior product analyst explaining why a metric changed between two periods.
You receive the metric value in both periods and its breakdown by segments of several dimensions
(<facts>), computed by pandas. For additive metrics, share_of_total_change_pct shows how much of the total
change each segment explains.

<task>
- headline: one sentence with the metric, the size of the change and the main driver.
- main_drivers: 2-5 segments that explain most of the change, with numbers from the breakdown.
- cause_hypotheses: 2-4 plausible business or technical causes that fit the pattern (concentrated in one segment
  vs. spread evenly across all segments points to different causes). Mark them clearly as hypotheses.
- next_checks: concrete checks that would confirm or reject the hypotheses.
</task>
+ COMMON_RULES
```

Розклад зміни повністю рахує pandas (`compute_change`): значення метрики в обох періодах, і для кожного сегмента – зміна та її частка в загальній зміні. Модель отримує вже готову відповідь на питання «де саме», а сама відповідає лише на питання «чому може бути».

Чому так:
- Пояснення поля `share_of_total_change_pct` прямо в промпті потрібне, щоб модель не плутала «сегмент впав на 64 %» (Android, мала база) з «сегмент дав 64 % падіння».
- Підказка «concentrated in one segment vs. spread evenly» – аналітична евристика: падіння в усіх сегментах вказує на загальну причину (сезонність, ціна, реліз), а падіння в одному – на локальну (канал, країна, збій).
- Вимога позначати причини як гіпотези відокремлює факти від припущень.

Результат: падіння виручки на 15,96 % пояснено сегментом app = mobile (85,73 % зміни) та групою середніх ринків; перевірка 88 з 88.

## 5. Executive summary (`SUMMARY_SYSTEM`)

```
You write an executive summary for a busy stakeholder (CEO, product lead) based on computed
metrics (<facts>). The reader has two minutes.

<format>
- title: the main message in one sentence (not the topic).
- what_happened: 2-4 bullets with the key numbers.
- why_it_matters: 1-3 bullets on business impact.
- next_steps: 2-4 concrete actions with an owner type (product, marketing, analytics).
Plain business language, no jargon, no hedging filler.
</format>
+ COMMON_RULES
```

Чому так. Схема «що сталося → чому важливо → що робити» відповідає класичній структурі бізнес-комунікації. «Title = main message, not the topic» забороняє заголовки на кшталт «Звіт про виручку». Обмеження кількості пунктів тримає текст коротким. Власник дії (product, marketing, analytics) робить наступні кроки придатними до виконання. Результат: перевірка 41 з 41 чисел.

## Перевірка чисел (`verify.py`) і її ітерація

Модуль витягує з відповіді моделі всі числа (зокрема «28 636.64», «15,96», «-4570.49») і шукає кожне серед чисел у фактах з допуском 2 %. Допускаються також варіанти ×100 / ÷100 (частка проти відсотка), тисячі та мільйони. Малі цілі (≤ 12: «топ-5», «3 гіпотези») і роки не перевіряються.

Ітерація: перший запуск EDA показав 5 «неперевірених» значень 79.99, хоча вони є в даних. Причина: це число стояло всередині ключа словаря (`'plan=annual;amount=79.99': 108` у `top_values`), а верифікатор тоді читав лише значення. Після виправлення (числа витягуються і з ключів) той самий EDA дав 150 з 150. Висновок: верифікатор теж треба тестувати. Тепер його покривають офлайн-тести, зокрема тест на навмисно вигадане число.
