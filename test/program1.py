# WAP to convert the list of city names into the the integer numbers.[encoding of city names]

from sklearn.preprocessing import LabelEncoder
city=["mumbai","pune","nagpur","chennai"]

encoder=LabelEncoder()
encoded=encoder.fit_transform(city)
print(encoded)

decoded=encoder.inverse_transform([0,2])
print(decoded)