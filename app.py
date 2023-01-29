"""Hops flask middleware example"""
from flask import Flask
import ghhops_server as hs
import rhino3dm

import math
import re
import urllib.request
import collections
from collections import Counter
from collections import OrderedDict
import os
from os import path
import random

import numpy as np
import numpy.linalg

# import matplotlib
# import matplotlib.pyplot as plt

import pandas as pd
#import io

#import scipy
#import seaborn
#import sklearn

#from lxml import objectify

# register hops app as middleware
app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)


# flask app can be used for other stuff drectly
@app.route("/help")
def help():
    return "Welcome to Grashopper Hops for CPython!"

"""
import json
import sklearn as skl
import sklearn.linear_model as linm
import sklearn.cluster as cluster
import sklearn.neighbors as nb
import sklearn.neural_network as MLP
import sklearn.tree
import sklearn.svm
import sklearn.ensemble
"""


"""
███╗   ███╗ ██████╗███╗   ██╗███████╗███████╗██╗                   
████╗ ████║██╔════╝████╗  ██║██╔════╝██╔════╝██║                   
██╔████╔██║██║     ██╔██╗ ██║█████╗  █████╗  ██║                   
██║╚██╔╝██║██║     ██║╚██╗██║██╔══╝  ██╔══╝  ██║                   
██║ ╚═╝ ██║╚██████╗██║ ╚████║███████╗███████╗███████╗              
╚═╝     ╚═╝ ╚═════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝              
                                                                   
███████╗██╗  ██╗ █████╗ ███╗   ███╗██████╗ ██╗     ███████╗███████╗
██╔════╝╚██╗██╔╝██╔══██╗████╗ ████║██╔══██╗██║     ██╔════╝██╔════╝
█████╗   ╚███╔╝ ███████║██╔████╔██║██████╔╝██║     █████╗  ███████╗
██╔══╝   ██╔██╗ ██╔══██║██║╚██╔╝██║██╔═══╝ ██║     ██╔══╝  ╚════██║
███████╗██╔╝ ██╗██║  ██║██║ ╚═╝ ██║██║     ███████╗███████╗███████║
╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝╚══════╝  
"""

@hops.component(
    "/binmult",
    name="BinMult",
    description="BinMult",
    category="Math",
    subcategory="Math",
    inputs=[hs.HopsNumber("A"), hs.HopsNumber("B")],
    outputs=[hs.HopsNumber("Multiply")],
)
def BinaryMultiply(a: float, b: float):
    return a * b


@hops.component(
    "/add",
    name="Add",
    nickname="Add",
    description="Add numbers with CPython",
    category="Math",
    subcategory="Math",
    inputs=[
        hs.HopsNumber("A", "A", "First number"),
        hs.HopsNumber("B", "B", "Second number"),
    ],
    outputs=[hs.HopsNumber("Sum", "S", "A + B")]
)
def add(a: float, b: float):
    return a + b


@hops.component(
    "/pointat",
    name="PointAt",
    nickname="PtAt",
    description="Get point along curve",
    category="Rhino3dm",
    subcategory="Rhino3dm",
    icon="pointat.png",
    inputs=[
        hs.HopsCurve("Curve", "C", "Curve to evaluate"),
        hs.HopsNumber("t", "t", "Parameter on Curve to evaluate")
    ],
    outputs=[hs.HopsPoint("P", "P", "Point on curve at t")]
)
def pointat(curve: rhino3dm.Curve, t=0.0):
    return curve.PointAt(t)


@hops.component(
    "/srf4pt",
    name="4Point Surface",
    nickname="Srf4Pt",
    description="Create ruled surface from four points",
    category="Rhino3dm",
    subcategory="Rhino3dm",
    inputs=[
        hs.HopsPoint("Corner A", "A", "First corner"),
        hs.HopsPoint("Corner B", "B", "Second corner"),
        hs.HopsPoint("Corner C", "C", "Third corner"),
        hs.HopsPoint("Corner D", "D", "Fourth corner")
    ],
    outputs=[hs.HopsSurface("Surface", "S", "Resulting surface")]
)
def ruled_surface(a: rhino3dm.Point3d,
                  b: rhino3dm.Point3d,
                  c: rhino3dm.Point3d,
                  d: rhino3dm.Point3d):
    edge1 = rhino3dm.LineCurve(a, b)
    edge2 = rhino3dm.LineCurve(c, d)
    return rhino3dm.NurbsSurface.CreateRuledSurface(edge1, edge2)


@hops.component(
    "/curve_end_points",
    name="EndPoints",
    nickname="EndPoints",
    description="Get curve start/end points",
    category="Rhino3dm",
    subcategory="Rhino3dm",
    #icon="beamupUserObjects/icons/bmd_level.png",
    inputs=[
        hs.HopsCurve("Curve", "C", "Curve to evaluate")
    ],
    outputs=[
        hs.HopsPoint("S"),
        hs.HopsPoint("E"),
        #hs.HopsNumber("EE", "EE", "test")
    ]
)
def end_points(curve: rhino3dm.Curve):
    start = curve.PointAt(0)
    end = curve.PointAt(1)
    return (end, start) #return (end, start, {"{0}": end.X, "{1}": start.X})

@hops.component(
    "/pointsat",
    name="PointsAt",
    nickname="PtsAt",
    description="Get points along curve",
    category="Rhino3dm",
    subcategory="Rhino3dm",
    icon="pointat.png",
    inputs=[
        hs.HopsCurve("Curve", "C", "Curve to evaluate"),
        hs.HopsNumber("t", "t", "Parameters on Curve to evaluate", hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsPoint("P", "P", "Points on curve at t")
    ]
)
def pointsat(curve, t):
    points = [curve.PointAt(item) for item in t]
    return points

#MACHINE LEARNING WITH CHATGPT AND PYTHON 3.0 IN GRASSHOPPER
#Write the python 3.0 script for the following psuedo code for the hops node in grasshopper for Rhino3d
#use flask and the ghhops_server in python 3.0 and the @hops.component
#create a machine learning algorithm for linear regression
#use the hops node to create a linear regression model
@hops.component(
    "/linear_regression",
    name="Linear Regression",
    nickname="Linear Regression",
    description="Linear Regression",
    category="Machine Learning",
    subcategory="Machine Learning",
    #icon="linear_regression.png",
    inputs=[
        hs.HopsString("X", "X", "X", hs.HopsParamAccess.LIST),
        hs.HopsString("y", "y", "y", hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsString("model", "model", "model")
    ]
)
def linear_regression(x, y):

    #Import the libraries
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    #Create the dataframe
    df = pd.DataFrame({'X': x, 'y': y})

    #Split the data into features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    #Create and fit the linear regression model
    lm = LinearRegression()
    lm.fit(X, y)
    
    #Print the type of the model
    print(type(lm))

    #convert the dataframe to a list
    data = df.values.tolist()

    #predict and model
    model = lm.predict(X).tolist()

    #return the model
    return model

@hops.component(
    "/appleStocks_LR",
    name="Apple Stocks LR",
    nickname="Apple Stocks LR",
    description="Apple Stocks LR",
    inputs=
    [
        hs.HopsString("x", "x", "x", hs.HopsParamAccess.LIST),
        hs.HopsString("y", "y", "y", hs.HopsParamAccess.LIST),
        hs.HopsString("z", "z", "z", hs.HopsParamAccess.LIST),
    ],
    outputs=
    [
        hs.HopsString("model", "model", "model")
    ]
)
def appleStocks_LR(x, y, z):
    #Import the libraries
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    
    # Data (Apple stock price)
    # https://finance.yahoo.com/quote/AAPL/history?p=AAPL

    #apple = pd.read_csv('AAPL.csv')
    apple = np.array([x, y, z])
    n = len(apple)

    #One-Liner
    model = LinearRegression().fit(np.arange(n).reshape(n,1), apple)
    print(model.predict([[3], [4], [5]]))
    print(type(model))

    #Result
    return model.predict([[3], [4], [5]]).tolist()




if __name__ == "__main__":
    app.run(debug=True)



