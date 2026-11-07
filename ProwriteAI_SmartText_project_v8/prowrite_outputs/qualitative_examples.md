### Example 1 (success (all models good))

| | text |
|---|---|
| Input | Hi everyone, could you send me address? |
| Reference | Hi everyone, could you send me the address? |
| A_LSTM | Hi everyone, could you send me the tickets tomorrow? *(chrF 74.4; ok)* |
| B_BiLSTM | Hi everyone, could you send me the tickets tomorrow? *(chrF 74.4; ok)* |
| C_Attention | Hi everyone, could you send me the address? *(chrF 100.0; ok)* |
| D_Pointer | Hi everyone, could you send me the address? *(chrF 100.0; ok)* |

### Example 2 (success (all models good))

| | text |
|---|---|
| Input | hi sir can you send me the address |
| Reference | Hello, could you send me the address? |
| A_LSTM | Hello, could you send me the tickets tomorrow? *(chrF 70.1; ok)* |
| B_BiLSTM | Hello, could you send me the tickets? *(chrF 74.0; ok)* |
| C_Attention | Hello, could you send me the address? *(chrF 100.0; ok)* |
| D_Pointer | Hello, could you send me the address? *(chrF 100.0; ok)* |

### Example 3 (success (all models good))

| | text |
|---|---|
| Input | hi sir - could you send me address? |
| Reference | Hello, could you send me the address? |
| A_LSTM | Hello, could you send me the tickets tomorrow? *(chrF 70.1; ok)* |
| B_BiLSTM | Hello, could you send me the tickets tomorrow? *(chrF 70.1; ok)* |
| C_Attention | Hello, could you send me the address? *(chrF 100.0; ok)* |
| D_Pointer | Hello, could you send me the address? *(chrF 100.0; ok)* |

### Example 4 (success (all models good))

| | text |
|---|---|
| Input | Hi could you send me teh address? |
| Reference | Hi, could you send me the address? |
| A_LSTM | Hi, could you send me the tickets tomorrow? *(chrF 66.7; ok)* |
| B_BiLSTM | Hi, could you send me the tickets tonight? *(chrF 66.6; ok)* |
| C_Attention | Hi, could you send me the address? *(chrF 100.0; ok)* |
| D_Pointer | Hi, could you send me the address? *(chrF 100.0; ok)* |

### Example 5 (models disagree)

| | text |
|---|---|
| Input | They will bring a cake next month |
| Reference | They will bring a cake next month. |
| A_LSTM | We will meet in Room 2A. *(chrF 13.0; new-number)* |
| B_BiLSTM | Please send me a bike today. *(chrF 10.5; ok)* |
| C_Attention | They will bring a cake next month. *(chrF 100.0; ok)* |
| D_Pointer | They will bring a cake next month. *(chrF 100.0; ok)* |

### Example 6 (models disagree)

| | text |
|---|---|
| Input | She will finish my homework on jun 15 |
| Reference | She will finish my homework on June 15. |
| A_LSTM | She usually washes the new on Thursdays. *(chrF 15.1; lost:15)* |
| B_BiLSTM | She usually washes the on Thursdays on Thursdays. *(chrF 12.4; lost:15)* |
| C_Attention | She will finish my homework on June 15. *(chrF 100.0; ok)* |
| D_Pointer | She will finish my homework on June 15. *(chrF 100.0; ok)* |

### Example 7 (models disagree)

| | text |
|---|---|
| Input | grace will change the tires this weekend |
| Reference | Grace will change the tires this weekend. |
| A_LSTM | Will you bring the new next month? *(chrF 13.6; ok)* |
| B_BiLSTM | Isabel will not happen again next week. *(chrF 18.2; negation-mismatch)* |
| C_Attention | Grace will change the tires this weekend. *(chrF 100.0; ok)* |
| D_Pointer | Grace will change the tires this weekend. *(chrF 100.0; ok)* |

### Example 8 (models disagree)

| | text |
|---|---|
| Input | he will pick up my sister next month |
| Reference | He will pick up my sister next month. |
| A_LSTM | She will be presented minutes late. *(chrF 18.8; ok)* |
| B_BiLSTM | He usually washes my account this weekend. *(chrF 15.1; ok)* |
| C_Attention | He will pick up my sister next month. *(chrF 100.0; ok)* |
| D_Pointer | He will pick up my sister next month. *(chrF 100.0; ok)* |

### Example 9 (failure (all models struggle))

| | text |
|---|---|
| Input | is it ok if i clean fridge on mar 14 |
| Reference | Is it okay if I clean the fridge on March 14? |
| A_LSTM | It is too rainy to go running today. I am worried about the. *(chrF 12.9; lost:14)* |
| B_BiLSTM | It is too hot to go to the dentist on Tuesday. *(chrF 12.4; lost:14)* |
| C_Attention | It will it that works if that I clean on Friday. *(chrF 24.0; lost:14)* |
| D_Pointer | She it has if anything I need to fridge to mar 14. *(chrF 24.8; ok)* |

### Example 10 (failure (all models struggle))

| | text |
|---|---|
| Input | Main webste is down rn!! |
| Reference | The main website is currently down. |
| A_LSTM | The weekly team is already working again. *(chrF 19.6; ok)* |
| B_BiLSTM | The hotel is near the station. *(chrF 14.5; ok)* |
| C_Attention | The approval process is feeding the right now. *(chrF 16.4; ok)* |
| D_Pointer | The webste is now down right now. *(chrF 24.5; ok)* |

### Example 11 (failure (all models struggle))

| | text |
|---|---|
| Input | should only take like an hour. anna and i understood it differently |
| Reference | It should only take about an hour. Anna and I understood it differently. |
| A_LSTM | Sorry, I forgot to open the new. I am working from home tomorrow. *(chrF 11.1; ok)* |
| B_BiLSTM | I would like to take a short extension on my application. *(chrF 17.6; ok)* |
| C_Attention | Should would like to an umbrella? I will cover the work, and I regret it. *(chrF 17.5; ok)* |
| D_Pointer | What only to like an hour? Anna has and, but I understood it. *(chrF 39.5; ok)* |

### Example 12 (failure (all models struggle))

| | text |
|---|---|
| Input | online is also ok if easier for you |
| Reference | I can also meet online if that is easier for you. |
| A_LSTM | There is a problem with the invoice. Thank you. *(chrF 17.7; ok)* |
| B_BiLSTM | The system says my desk has been out of toner for a week. *(chrF 20.2; ok)* |
| C_Attention | Are is a problem if anything. Thank you for you. *(chrF 23.8; ok)* |
| D_Pointer | The is also improved ok if easier if you. *(chrF 25.9; ok)* |

