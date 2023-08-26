from flask import Flask , render_template, request , jsonify, session

# Create a Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
# Define a route and function to handle requests to that route

var="hakka"


@app.route('/', methods=['GET', 'POST'])
def hello_world():



################################################################ axios work ####
    final_answer='shown after calculation'

    if request.method == 'POST':

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        from sklearn.model_selection import train_test_split

        f = pd.read_csv("Bengaluru_House_Data.csv")
        data = pd.DataFrame(f)

        data["location"] = data["location"].fillna("Sarjapur Road")

        data["size"] = data["size"].fillna("2 BHK")
        data["bath"] = data["bath"].fillna(data["bath"].median())
        data['balcony'] = data['balcony'].fillna(data['balcony'].median())

        ########################################### dealing with taotal_sqft ########################

        for index, row in data.iterrows():
            temp = row["total_sqft"].split('-')
            if len(temp) == 2:
                data.loc[index, "total_sqft"] = (float(temp[0]) + float(temp[1])) / 2
            else:
                try:
                    data.loc[index, "total_sqft"] = float(temp[0])
                except Exception as e:
                    data = data.drop(index=index)

        data = data.reset_index(drop=True)

        #############################################################################################

        new_data = data.copy()

        for i in range(len(new_data["size"])):
            temp = new_data.loc[i, "size"].split()
            new_data.loc[i, "bhk"] = float(temp[0])

        data = pd.concat([data, new_data["bhk"]], axis=1)

        ####################### price per sq feet #####################

        data['price per sq feet'] = (data['price'] * 100000) / (data['total_sqft'])
        data['price per sq feet'] = data['price per sq feet'].astype(float)

        data = data.drop(['availability', 'society', 'size'], axis=1)

        ##########################  ML work     #####################################

        sample_data = data.iloc[:, [0, 1, 6, 3, 4, 2]]

        x = sample_data.iloc[:, 0:5].values
        y = sample_data.iloc[:, -1].values.reshape(-1, 1)

        print("sample_data->", sample_data.columns)

        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder(drop='first', sparse_output=False)
        x = ohe.fit_transform(x[:, [0, 1]])

        x = np.hstack((x, sample_data.iloc[:, 2:-1].values))

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=43)

        #print("method post")
        data1 = request.json
        area_type = data1.get('area_type')
        location = data1.get('location')
        bhk = data1.get('bhk')
        bath = data1.get('bath')
        balcony = data1.get('balcony')
        #print("form->", area_type, location, bhk, bath, balcony)
        #print(data)

        list_user=[area_type,location,bhk,bath,balcony]

        #############################################################################################################
        #user_data = ['Super built-up  Area', 'Lingadheeranahalli', '3', '3', '1']
        #############################################################################################################
        user_data=list_user
        user_frame = pd.DataFrame([user_data], columns=['area_type', 'location', 'bhk', 'bath', 'balcony'])
        user_x = user_frame.iloc[:, 0:5].values
        mid = ohe.transform(user_x[:, [0, 1]])
        user_x = np.hstack((mid, user_frame.iloc[:, 2:5]))
        print(user_x)

        ################## DECISION TREE FOR TOTAL SQFT ####################

        from sklearn.tree import DecisionTreeRegressor
        tree = DecisionTreeRegressor()
        tree.fit(x_train, y_train)
        total_sqft_pred = tree.predict(x_test)

        user_sqft_pred = tree.predict(user_x)

        user_sqft_pred = user_sqft_pred[0]

        # print("user_sqft_pred->",user_sqft_pred)

        #####################  price prediction  ###############################################

        # user_data=['Super built-up  Area','Electronic City Phase II','3','3','1']
        # user_frame=pd.DataFrame([user_data],columns=['area_type','location','bhk','bath','balcony'])

        sample_data2 = data.iloc[:, [0, 1, 2, 5]]

        print("sample_data2->", sample_data2.columns)
        # print(sample_data2.columns)
        x2 = sample_data2.iloc[:, :-1].values
        y2 = sample_data2.iloc[:, [3]].values.reshape(-1, 1)

        user_price_2 = pd.DataFrame({'area_type': user_data[0], 'location': user_data[1], 'total_sqft': user_sqft_pred},
                                    index=[0]).values

        ohe2 = OneHotEncoder(drop='first', sparse_output=False)
        mid2 = ohe2.fit_transform(x2[:, [0, 1]])
        x2 = np.hstack((mid2, sample_data2.iloc[:, 2].values.reshape(-1, 1)))

        user_mid = ohe2.transform(user_price_2[:, [0, 1]])

        user_price_2 = np.hstack((user_mid, user_price_2[:, [2]]))
        print(user_price_2)

        ################## another D_Tree for price prediction ###########################

        price_tree = DecisionTreeRegressor()

        x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.3, random_state=0)

        price_tree.fit(x_train2, y_train2)

        price_pred = price_tree.predict(x_test2)

        user_price_pred = price_tree.predict(user_price_2)[0]

        print("final_predicted_valueeee", user_price_pred)
        final_answer=user_price_pred

        print("final_answer",final_answer)

        var=final_answer
        #print(var)
        session['final_answer'] = final_answer
        return render_template('index2.html',final_answer=final_answer)

    else:

        return render_template('index2.html',final_answer=final_answer)



@app.route('/get_data', methods=['GET'])
def get_data():
    final_answer = session.get('final_answer', 'No data available')
    final_data = {"predicted_price": final_answer}
    return jsonify(final_data)









# Run the app if this script is executed
if __name__ == '__main__':
    app.run()
