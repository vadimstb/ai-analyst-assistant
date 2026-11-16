# Питання: Порахуй retention 7-го дня для когорт користувачів за тижнем реєстрації

## Згенерований код
```python
d = df.copy()
d['reg_day'] = d['reg_dt'].dt.normalize()
d['pay_day'] = d['pay_dt'].dt.normalize()
d['day_n'] = (d['pay_day'] - d['reg_day']).dt.days

start = d['pay_day'].min()
end = d['pay_day'].max()

users = d.groupby('user_id')['reg_day'].first()
# only cohorts registered inside the observation window that have a full 7 days of follow-up
users = users[(users >= start) & (users + pd.Timedelta(days=7) <= end)]

cohort_week = users.dt.to_period('W').dt.start_time
retained_ids = set(d.loc[d['day_n'] == 7, 'user_id'])

u = pd.DataFrame({'cohort_week': cohort_week})
u['retained'] = u.index.isin(retained_ids)

result = (u.groupby('cohort_week')
            .agg(cohort_size=('retained', 'size'), retained_d7=('retained', 'sum'))
            .reset_index())
result['retention_d7_pct'] = (result['retained_d7'] / result['cohort_size'] * 100).round(2)
result = result.sort_values('cohort_week').reset_index(drop=True)
```

Код рахує retention 7-го дня для когорт за тижнем реєстрації (тиждень починається з понеділка). Користувач вважається утриманим, якщо зробив платіж рівно на 7-й календарний день після реєстрації.

Припущення:
- **Що вважається активністю.** У даних є лише платежі, тому активністю вважається платіж. Через це в когорти потрапляють тільки користувачі, які хоч раз платили.
- **Які когорти враховано.** Беру лише користувачів, зареєстрованих у межах періоду, за який є платежі (з 5 квітня по 11 травня 2024). Також у кожного має бути повний 7-денний період після реєстрації. Старіші реєстрації (з 2015 року) не враховано.

## Результат

| cohort_week         |   cohort_size |   retained_d7 |   retention_d7_pct |
|:--------------------|--------------:|--------------:|-------------------:|
| 2024-04-01 00:00:00 |           232 |            18 |               7.76 |
| 2024-04-08 00:00:00 |           494 |            33 |               6.68 |
| 2024-04-15 00:00:00 |          1060 |            36 |               3.4  |
| 2024-04-22 00:00:00 |          1681 |            18 |               1.07 |
| 2024-04-29 00:00:00 |          1587 |            12 |               0.76 |
