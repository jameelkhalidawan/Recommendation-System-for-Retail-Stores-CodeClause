# Recommendation-System-for-Retail-Stores-CodeClause
This is a Python script for building a Retail Store Recommendation System. The system incorporates the following components:
Data preprocessing and analysis
Collaborative filtering using a Neural Collaborative Filtering (NCF) model
Product recommendations based on user input (state)
A basic graphical user interface (GUI) using Tkinter for user interaction

**Prerequisites**

Ensure you have Python 3.x installed on your system.
Install the required libraries by running the following command:

pip install pandas numpy torch scikit-learn matplotlib

**Running the Code**

Clone this GitHub repository or download the script (retail_store_recommendation.py) to your local machine.

Prepare the data:

Replace 'data.csv' with your dataset file.
Ensure the dataset contains columns named 'Customer_ID', 'Product_ID', 'Sales', 'State', and 'Product_Name'.
Open a terminal or command prompt and navigate to the directory where the script is located.

Run the script using the following command:
python retail_store_recommendation.py

A GUI window will open. You can enter a state in the text input field and click the "Recommend Products" button to receive product recommendations based on the entered state.

You can also click the "Graphs & Insights" button to view additional data visualizations (e.g., a histogram of sales distribution).

**Model Hyperparameter Tuning**

The script also includes a hyperparameter tuning section where it searches for the best hyperparameters for the NCF model using a random search approach. The best hyperparameters are displayed at the end of the script.

**Additional Information**

The script utilizes the TF-IDF vectorization technique to compute the cosine similarity between products.
The NCF model is used for collaborative filtering to recommend products to users.
The GUI is built using the Tkinter library for a basic user interface.
Feel free to customize the code and adapt it to your specific dataset and requirements.

If you encounter any issues or have questions, please don't hesitate to contact me.

Enjoy using the Retail Store Recommendation System!








