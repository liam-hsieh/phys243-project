## House Improvement ROI Prediction Model

#### Group 4
#### James Siefert, Liam Hsieh, Panawannage Fernando, and Christopher Moussa

---

### Background

The housing market plays a crucial role in today’s national economy,
representing one of the most significant sectors in terms of financial
transactions and investment opportunities. The valuation and prediction of
housing prices have garnered significant attention from researchers,
policymakers, and real estate professionals, as it directly impacts numerous
stakeholders, including homeowners, buyers, sellers, investors, and government
institutions.

Being able to predict housing prices based on a number of factors using
historical data can effectively capture intricate relationships and identify
crucical factors influencing housing prices. In this project, we separate and
define these factors into two categories: **changeable features** (square
footage, number of bedrooms and bathrooms, number of stories, presence of a
guest room, central air conditioning, and furnishing status) and
**non-changeable features** (proximity to a main road, presence of a basement,
hot water heating, parking spaces, and preferred area status). These factors,
when analyzed, can help buyers make informed decisions.

### Implementation

This project looks to investigate the value increase associated with a number
of home improvement projects to help homeowners maximize their **return on**
**investment (ROI)**. It uses a [Kaggle](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)
dataset along with a number of trusted articles to guage values and other
necessary information regarding home features and their relation to overall
home price.

This project is broken into multiple phases: module creation, model training,
model evaluation and refinement, and ROI calculation and integration. Each phase
builds off of the previous in the Jupyter notebook to describe the machine learning
process for the housing dataset, extensions of the data provided in the dataset
through the creation of Python modules, and web appplication interfaces that link
the trained model with the Python modules.

### Goals

The goal of this project is to extend beyond the basic machine learning process
with sample data and challenge ourselves to think about an extensible problem.
While this project takes a number of steps past creating a linear regression model
for the Kaggle dataset, it can be extended and advanced with the addition of more
parameters and more data and training.

### How to Run the Web UI locally
* Step 1: Go to `web_app/` -> `web_app.py`
* Step 2: Run `web_app.py`
* Step 3: Go to `http://127.0.0.1:5000` in your web browser
* Step 4: Alternate: Go into the `project.ipynb` file to view the application after `web_app.py` is running
* Step 5: Alternate 2: Contact James Siefert (james.siefert@email.ucr.edu) and I can run a local server and send a URL for easy access.

### How to Run the Web UI locally in a virtual environment in Windows and Python 3.10.9
* Step 1: open command prompt ("cmd" in search bar)
* Step 2: ensure you are using `3.10.9` by running `python --version`
* Step 3: navigate to project directory with `cd <project folder path>`
* Step 4: run `python -m venv venv`
* Step 5: run `venv\Scripts\activate`
* Step 6: install requirements: `pip install -r requirements.txt`
* Step 7: change directories to the `web_app/` directory: `cd web_app`
* Step 8: run the web app: `python web_app.py`
* Step 9: Go to `http://127.0.0.1:5000` in your web browser
