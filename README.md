#Predictive Model to enhance Student placement opportunities through skill-based self-assessment Using ML

ABSTRACT

The Machine Learning based Predictive model to enhance the placement opportunities of  the  students studying engineering degree. The model takes the reference data  from  the  previously  placed students under the same stream. The proposed predictive model makes use of the Machine Learning Algorithm to assess the student skills based on the current performance. The model also guide to student to improve their placement records by recommending various skill enhancement programmes and training.
Keywords: Placement, Skill Assessment, Machine Learning, Improvement etc.,

INTRODUCTION

After making various recordings and studies stats were made showing 1.5 million engineering  students graduate each year in India. The demand  and need for well qualified engineering graduates  in IT industries and analytical industries are rising day in and day. But most of the students who are graduating from engineering colleges are oblivious to the needs to these industries. The no.  of students who are good enough for these roles are very low in comparison to the total no of students graduating.   It is an universally known fact all around the globe that a reputation of a university and   a college depends on the placements it can provide to its students.
Each and every institutions and universities (most of them) have separate cells called the placement cell dedicated to this purpose only who strive year after year to attain maximum if not 100% placement for all their students’ .Each college has their previous years student record and placement data. This information’s are a treasure trove of insights and data, but its huge for any one student to decipher it and make a good use out of it .

 
Finding meaning and making predictions and insights from these data is an essential part of educational research. For this process not only understanding of the data is necessary but a proper understanding of various components as in : Knowledge possessed by the students , Skills acquired by them in 3 years and attitude which is necessary to work in an corporate environment is equally important and this cannot be assessed by student on his own. Hence a need for a system or algorithm was felt to help and guide students in this process.

PROBLEM STATEMENT

Each and every student dreams of a successful career and job at the best companies where they can grow and contribute to and work with some of the best and upcoming technologies and concepts. However because of lack of knowledge of how a placement process works or how to get hired by his/her dream company, an undergraduate falls short of preparation and hence fails to crack into to the screening and interview process. As an undergraduate I exactly know how it feels to prepare for a company for night and day and still fall short. Depression, anxiety and self doubt sets in . On top of that reluctance and carelessness from the companies and the college (as well) in informing students the positives and negatives of their performance in their exam makes the situation even more grim for the student. They feel lost and dejected. Hence a need for a system or algorithm was felt to help and guide students in this process and help students to get placements in good companies and be proud of themselves.

MODULE DESCRIPTION

The entire process can be decided in 3 sections or modules
1)	Collection and cleaning of student data from sources and making data appropriate and ready to be worked on by various machine learning algorithms and data mining techniques . This process is called data pre-processing and it is the very first step to the entire process.
There are python libraries for the process.
2)	Predicting the probability of the student of getting placed with his/her present knowledge, skill and aptitude. For this various models like k- neatest neighbour’s classification and other same or like models as that of logistic regression is used.
3)	Prediction of eligibility of a student and ways to improve and levels of action required by them to crack through the screening process. This is achieved using a combination of Fuzzy approach and methods of Rule based classification while prediction using decision tree algorithm shows the qualities a student needs to improve in order to crack the interviews.
 
PROCEDURES AND EXPLANATIONS

1)	DATA ACQUIRING

1.A) DATA COLLECTION

The data used in this paper is collected from the statistical department of SRM Institute of Science and Technology Kattankulathur, Tamil Nadu.
Data used in this paper is the previous year’s student data of placements as in the company they got into, tier of the company and the package provided as CTC in those companies. It also consists of the position the company came for recruitment in the college. Data of students on the marks and grade obtained and skills acquired throughout their three years in college is also considered and recorded. Eleven and twelve standard marks, sex and place of birth are also taken into account.
It has been observed that factors like communication, Analytical process, teamwork and observational skills are considered as an important parameter in the process of selection and hiring. However due to the lack of availability of historical or training data such parameters are discarded for now but may be considered in the post data processing process. For the data to be considered, past 8 year’s student record are taken into account .

1.B) DATA PRE-PROCESSING

Pre-processing information is that the start to begin the activity. Typically, procured information is temperamental, conflicting, wrong (contains mistakes or anomalies) and conjointly deficient with regards to clear values and patterns. This is visible where pre-preparing information joins things

◦	it assists with cleaning, designing, and orchestrate crude information.


Python libraries utilized for this knowledge preprocessing in Machine Learning are:


◦	Numpy: NumPy is an essential bundle for mathematical calculation present Python. Used to fuse variety of numerical cycle into the code. It allows one conjointly apply gigantic 4D arrays and grids to the code exploitation NumPy.

◦	Pandas: Pandas , mainstream open – source library for preparing and examination. it's ordinarily utilized for the import and support of knowledge sets or data set. It's stuffed with superior, simple to-utilize information models, information handling assets for Python and much more.
 
◦	Scikit: Scikit-learn is ML based library available in Python. It highlights many algorithm such as support, random woods, and k-neighbors, and also sup- ports Python computational and science libraries such as NumPy and SciPy.

Matplotlib: Matplotlib is a Python library used for plotting 2D maps . It will provide publication- quality data in a range of hard copy formats and immersive environments across networks (IPython shells, Jupyter notebooks, online application servers,etc.).

2)	PREDICTING THE PROBABILITY OF A STUDENT GETTING PLACED
We  make use of the K-nearest  neighbor classification as the model that  form the basis of the 2nd   step of the module . This model would help us to determine the probability of the student acquiring    a position as a fresher in a company.
The two class would be taken into consideration are yes and no. The outcome would be either an yes or a no. Then the result would be compared to other binary model of clarification like that of Logistic Regression and . The process mentioned above are implemented in python 2.7[7].


2.A)	K-NEAREST NEIGHBORING CLASSIFICATION
It is an algorithm that uses supervised learning and the classification for this is done on the basis of the distance calculated midst of the training data set and testing data set . This distance is measured using Euclidean Distance.
The similarities between the training and testing data set the k neighbors are assigned. Here k is (+) in value. Testing data is associated to the class that got maximum no of votes among the k values of the nearest neighbor.
K-closest neighbours' basic algorithm
1)	Value for k is found out.
2)	The separation between each of the records of the preparation set and the testing record is registered.
3)	Neighbours are sorted in the expanding request of the separations.
4)	Principal k neighbours are selected from the arranged rundown.
5)	The class to which majority of the neighbors belong to is located and assigned to the training data.
 
 
Fig 1.1 k- nearest neighbours classifier



2.B)	EUCLIDEAN DISTANCE

Euclidean separation is utilized to quantify separation between two focuses in euclidean space. This measured separation could be utilized as a proportion of closeness between the considered focuses.
                   	(1)
2.C)	LOGISTIC REGRESSION

This regression model is a method of statistics that is used to analyse one ore more than one
independent variable that is instrumental in determining the the outcome . The outcome so found is measured against dichotomous variable.
Logistic regression is chosen because this function takes in input , that is of any value positive or negative and  gives an output that is between one and zero. Hence its output could be considered as    a probability. 0 being minimum and 1 being maximum.


The paper[3] makes a proposition of a placement analyzing system that makes recommendation on
 
best placements for a student on placement is predicted is considered
 
the basis of their current capabilities. Five
 
separate tires of
 
•	Dream Companies (CTC >= 10 LPA)
•	Core Companies(CTC >= 4.5 LPA <=10 LPA )
•	Mass Recruitment (CTC<= 4.5 LPA)
•	Not Eligible.
 
The prediction in the above mentioned paper is done using logistic regression. The data set for the work includes basic student details like their marks, gender , college grades etc.

3) PREDICTION OF ELIGIBILITY OF STUDENTS AND WAYS TO IMPROVE

This section Predicts the eligibility of a student finds ways to improve and casts light on levels of
action  required  by them  to  crack  through the  screening process.  All  these are   achieved  using   a
combination of Fuzzy approach and methods of Rule based classification while prediction using
 
decision tree algorithm shows the interviews.
 
qualities a student needs to improve in order to crack the
 

3. A) FORECASTING PROBABILITY UTILISING FUZZY APPROACH

Paper[4] puts forward a system that would predict the eligibility and the actions and to be taken by a student in improving the overall eligibility in the campus placement using the combination of Fuzzy approach and Rule based methods of classification. The combination of the above  mentioned  methods is applied on the student’s testing dataset  containing academic and placement  information  to predict outcome. Here the attributes tale linguistic values that is “HIGH” or “LOW” to indicate where the student stands in comparison to other people who have got job from the training data set .
Set of rules are constructed from the database using Rule based classification. The rules are
evaluated to understand steps that a student can take to become ready for the placement season.


3.B)FORECASTING PROBABILITY UTILISING DECISION TREE ALGORITHM

Paper[5] puts forward a model that would predict the placement probability using ID3 decision tree
 
algorithm . The above mentioned
 
model analyses the provided data set and
 
selects the most
 
important /relevant attribute from the student data set. The entropy of the give data set and
Information gain value of all the attributes are measured and attributes having sufficient value is selected constructing the ID3 decision tree at the same time. The Weka tool is used in generating a optimised decision tree , whose leaves represent the student’s placement chances.

The formula used for entropy estimation is:
 
The equation for gain in information as the difference calculated between the given dataset and entropy of sub-divided data set is:



DATA OVER VIEW AND IT IS STRUCTURED

Data  that  is  used  for  this  paper  is  collected  from  the  statistic  department  of  the  SRM  IST Ktr
campus. From the collected data t  e parameters for the placement which has an history of being
highly influential is considered. The following table depicts some. Variables that are selected for making classification:
Table 1.1 Sample Data set

Variable	Range	Type
Sex	0,1	Number
Tenth standard agg.	0-100	Number ,cont.
Twelve stadard agg.	0-100	Number , cont.
B tech agg.	0-100	Number , cont
Backlogs	0,1	Number
Prediction	0-100	Number ,cont

There are some more skills which are also necessary for doing well in a placement season , like that  of skills pertaining to communicating with interviewers and pear,  technical  mastery and how skilful  a student is , team work etc.  However  as there were no  available historical data related to  such  skills therefore they were considered in post pre-processing stage.
Table 1.2 Post processing stage skills

Variable	Range	type
Tech	0-10	Number
communicating	0-10	Number
teamwork	0-10	Number
analysis	0-10	Number


TESTING AND RESULT

 
The test for the eligibility is conducted using the student data records of our previous years. The entire set is divided into two parts
 
college from the
 
training student data set and test student data set. 80% of the overall data is set for training set while the rest 20 % is set for the test set.
As the first step of the process the training data is given to the k-nearest algo. The result fro this is integrated with the rating of the skill set that these students have . Both the results are have 50% weight of the overall prediction process and the outcome comes out to be a value , either 1 or 0 and the proposed accuracy is (78%) [3].
When this result is obtained it is compare other binary classification models used that are logistic regression. And this would give a accuracy percentage around 75% [3].
after the prediction of eligibility process gets over then we would use fuzzy approach and rule based classification method to determine the value of the attribute for the student (high or low) followed by prediction using discussion tree to knit pick the parameters that a student should work on to improve his/her overall score .

CONCLUSION

This was an attempt to predict the eligibility probability of a student during a placement season and trying to determine areas that a student can improve to improve his/her chances in placement. Models like k-nearest neighbour is logistic regression is used to predict placement and further fuzzy approach and rule based classification method to determine the value of the attribute for the student and prediction using decision tree to knit pick the parameters that a student should work on to improve his/her overall score. This information would not only help students to strategize but placement cells to improve its process.


