from pickle import load
# load the model
model = load(open('model.pkl', 'rb'))
# load the scaler
scaler = load(open('scaler.pkl', 'rb'))
#predict user age 20 and salary 900000
user_age_salary=[[20,900000]]
scaled_result = scaler.transform(user_age_salary)
res=model.predict(scaled_result)
if res==1:
    print("He can buy the car")
else:
    print("He can't buy the car")