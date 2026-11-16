# Питання: Покажи топ-10 країн за виручкою у травні 2024

## Згенерований код
```python
mask = (df['pay_dt'] >= '2024-05-01') & (df['pay_dt'] < '2024-06-01')
result = (df[mask]
          .groupby('country_name', as_index=False)['amount_usd'].sum()
          .rename(columns={'country_name': 'Країна', 'amount_usd': 'Виручка, USD'})
          .sort_values('Виручка, USD', ascending=False)
          .head(10)
          .reset_index(drop=True))
result['Виручка, USD'] = result['Виручка, USD'].round(2)
```

Код рахує виручку як суму amount_usd за датою оплати (pay_dt) у травні 2024 року. Потім групує її за країнами й показує 10 країн із найбільшою сумою. У даних травень є лише до 11.05.2024, тому місяць неповний.

## Результат

| Країна                                               |   Виручка, USD |
|:-----------------------------------------------------|---------------:|
| Ukraine                                              |        9463.92 |
| United States of America                             |        7445.76 |
| South Africa                                         |        2618.95 |
| Spain                                                |        2436.69 |
| Chile                                                |        2294.81 |
| United Kingdom of Great Britain and Northern Ireland |        1790.54 |
| Argentina                                            |        1773.84 |
| Kazakhstan                                           |        1515.48 |
| Germany                                              |        1211.54 |
| Poland                                               |        1204.07 |
