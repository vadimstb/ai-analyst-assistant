# Питання: Порівняй середній чек і кількість платних користувачів по платформах (app) окремо для жінок і чоловіків

## Згенерований код
```python
d = df.copy()
d['gender'] = d['gender'].astype(str).str.strip().str.lower()
d['app'] = d['app'].astype(str).str.strip().str.lower()
result = (d.groupby(['gender', 'app'])
           .agg(avg_check_usd=('amount_usd', 'mean'),
                paying_users=('user_id', 'nunique'),
                payments=('amount_usd', 'size'),
                revenue_usd=('amount_usd', 'sum'))
           .reset_index())
result['avg_check_usd'] = result['avg_check_usd'].round(2)
result['revenue_usd'] = result['revenue_usd'].round(2)
result = result.rename(columns={'gender': 'Стать', 'app': 'Платформа', 'avg_check_usd': 'Середній чек, $',
                                'paying_users': 'Платних користувачів', 'payments': 'Кількість платежів',
                                'revenue_usd': 'Виручка, $'})
result = result.sort_values(['Стать', 'Середній чек, $'], ascending=[True, False]).reset_index(drop=True)
```

Код групує платежі за статтю та платформою (app). Для кожної групи рахує середній чек як середню суму одного платежу та кількість платних користувачів як кількість унікальних user_id. Для контексту додано також кількість платежів і виручку. Якщо користувач платив з одного профілю, його стать і платформа вважаються незмінними.

## Результат

| Стать   | Платформа   |   Середній чек, $ |   Платних користувачів |   Кількість платежів |   Виручка, $ |
|:--------|:------------|------------------:|-----------------------:|---------------------:|-------------:|
| f       | android     |             13.04 |                      9 |                   11 |       143.39 |
| f       | mobile      |              8.39 |                     45 |                  134 |      1124.48 |
| f       | desktop     |              7.65 |                      5 |                   11 |        84.13 |
| f       | ios         |              6.86 |                     41 |                   55 |       377.49 |
| m       | android     |             11    |                    144 |                  206 |      2265.78 |
| m       | desktop     |              8.76 |                    249 |                  367 |      3213.62 |
| m       | ios         |              8.69 |                    351 |                  514 |      4467.77 |
| m       | mobile      |              8.18 |                   6937 |                10032 |     82072.9  |
