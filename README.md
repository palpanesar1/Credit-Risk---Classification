# Credit-Risk---Classification

The Business Context
A major bank wants to better predict the likelihood of default for its customers, as well as identify the key
drivers that determine this likelihood. They hope that this would inform the bank’s decisions on who to
give a credit to and what credit limit to provide, as well as also help the bank have a better understanding
of their current and potential customers, which would inform their future strategy, including their
planning of offering targeted credit products to their customers.

The Data
The bank collected data on 25 000 of their existing clients. Of those, 1 000 were randomly selected to
participate in a pilot described below. Data about the remaining 24 000 is in the file “MMA867 A3 – credit
data.xls”. The dataset contains various information, including demographic factors, credit data, history of
payment, and bill statements from April to September, as well as information on the outcome: did the
customer default or not in October.

The screenshot below depicts the first 10 rows of the data:
ID LIMIT_BAL SEX EDUCATIO MARRIAGE AGE PAY_1 PAY_2 PAY_3 PAY_4 PAY_5 PAY_6 BILL_AMT1BILL_AMT2BILL_AMT3BILL_AMT4BILL_AMT5BILL_AMT6 PAY_AMT1 PAY_AMT2 PAY_AMT3 PAY_AMT4 PAY_AMT5 PAY_AMT6 default_0
1 20000 2 2 1 24 2 2 ‐1 ‐1 ‐2 ‐2 3913 3102 689 0 0 0 0 689 0 0 0 0 1
2 90000 2 2 2 34 0 0 0 0 0 0 29239 14027 13559 14331 14948 15549 1518 1500 1000 1000 1000 5000 0
3 50000 2 2 1 37 0 0 0 0 0 0 46990 48233 49291 28314 28959 29547 2000 2019 1200 1100 1069 1000 0
4 50000 1 2 1 57 ‐1 0 ‐1 0 0 0 8617 5670 35835 20940 19146 19131 2000 36681 10000 9000 689 679 0
5 50000 1 1 2 37 0 0 0 0 0 0 64400 57069 57608 19394 19619 20024 2500 1815 657 1000 1000 800 0
6 1.00E+05 2 2 2 23 0 ‐1 ‐1 0 0 ‐1 11876 380 601 221 ‐159 567 380 601 0 581 1687 1542 0
7 140000 2 3 1 28 0 0 2 0 0 0 11285 14096 12108 12211 11793 3719 3329 0 432 1000 1000 1000 0
8 20000 1 3 2 35 ‐2 ‐2 ‐2 ‐2 ‐1 ‐1 0 0 0 0 13007 13912 0 0 0 13007 1122 0 0
9 2.00E+05 2 3 2 34 0 0 2 0 0 ‐1 11073 9787 5535 2513 1828 3731 2306 12 50 300 3738 66 0
10 260000 2 1 2 51 ‐1 ‐1 ‐1 ‐1 ‐1 2 12261 21670 9966 8517 22287 13668 21818 9966 8583 22301 0 3640 0

Data Dictionary
 ID: ID of each client
 LIMIT_BAL: Total amount of credit line with the bank (including all individual and
family/supplementary credit)
 SEX: Gender (1=male, 2=female)
 EDUCATION: Education (1=graduate, 2=undergraduate, 3=high‐school, 4=other,
5,6=unknown)
 MARRIAGE: Marital status (1=married, 2=single, 3=other)
 AGE: Age in years
 PAY_1: Repayment status 1 month ago, – in September: (‐2=no need to pay, zero balance,
“payment holiday”, etc., ‐1=paid in full, 0=revolving credit (meaning client paid more than the
minimum payment, but less than the total balance), 1= delay for one month, ... 8=delay for 8
months, 9=delay for 9 months or more)
 PAY_2: Repayment status 2 months ago, – in August (scale as above for PAY_1)
 PAY_3: Repayment status 3 months ago (scale as above for PAY_1)
 PAY_4: Repayment status 4 months ago (scale as above for PAY_1)
 PAY_5: Repayment status 5 months ago (scale as above for PAY_1)
 PAY_6: Repayment status 6 months ago (scale as above for PAY_1)
 BILL_AMT1: Amount of bill statement 1 month ago, – in September
 BILL_AMT2: Amount of bill statement 2 months ago
Assignment 3, Due May 29 [FRI, for Sec A] / May 30 [SAT, for Sec B], 2019 at 11:59pm
MMA867: Predictive Modelling 2021W
 BILL_AMT3: Amount of bill statement 3 months ago
 BILL_AMT4: Amount of bill statement 4 months ago
 BILL_AMT5: Amount of bill statement 5 months ago
 BILL_AMT6: Amount of bill statement 6 months ago
 PAY_AMT1: Amount of payment 1 month ago, – in September
 PAY_AMT2: Amount of payment 2 months ago
 PAY_AMT3: Amount of payment 3 months ago
 PAY_AMT4: Amount of payment 4 months ago
 PAY_AMT5: Amount of payment 5 months ago
 PAY_AMT6: Amount of payment 6 months ago
 Default_0: Default in October (1=yes, 0=no)

The ultimate question: which of the 1 000 “new applicants” in the pilot should be issued credit?
In your analyses, please make the following simplifying assumptions:
 Defaults on the previously issued credit is not your problem
 All the clients who will be offered the credit line will use it in full
 Your cost of capital = 0
In other words, for each client in the pilot, if the credit is issued and repaid, then the bank earns a
profit of 25,000*2% + 1,000 = 1,500; if the credit is granted but the client defaults, then the bank
loses 25,000 ‐ 20,000 = 5,000? And if the credit is not issued, then the profit=loss=0.
