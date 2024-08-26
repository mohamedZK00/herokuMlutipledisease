import streamlit as st
from streamlit_option_menu import option_menu as om
import pandas as pd
import pickle 
from PIL import Image
import os

#model_Diabetes = pickle.load(open(r"E:\Pro_Stremlit بالعربي\Model Saving\Diabetes_prediction_RFR_83%.sav",'rb'))
#model_Heart = pickle.load(open(r"E:\Pro_Stremlit بالعربي\Model Saving\Heart_prediction_Vottingclf_90%.sav",'rb'))
#model_Parkinsons = pickle.load(open(r"E:\Pro_Stremlit بالعربي\Model Saving\Parkinsons_Prediction_Stacking_95%.sav",'rb'))


# Load the prediction model-1
working_dir1 = os.path.dirname(os.path.realpath(__file__))
model_path1 = os.path.join(working_dir1, 'models/Diabetes_prediction_RFR_83%.sav')
with open(model_path1, 'rb') as f:
    model_Diabetes = pickle.load(f)

# Load the prediction model-2
working_dir2 = os.path.dirname(os.path.realpath(__file__))
model_path2 = os.path.join(working_dir2, 'models/Heart_prediction_Vottingclf_90%.sav')
with open(model_path2, 'rb') as g:
    model_Heart = pickle.load(g)

# Load the prediction model-3
working_dir3 = os.path.dirname(os.path.realpath(__file__))
model_path3 = os.path.join(working_dir3, 'models/Parkinsons_Prediction_Stacking_95%.sav')
with open(model_path3, 'rb') as h:
    model_Parkinsons = pickle.load(h)







# sidabar for navigate
with st.sidebar:
    selected = om("Multiple Disease Prediction System",
                  ['Diabetes Prediction',
                   'Heart Disease Prediction',
                   'Parkinsons Prediction'],
                  
                    icons = ['activity','heart','person'],
                    
                   default_index = 0)
    

# Diabetes Prediction page
if(selected == 'Diabetes Prediction'):
    # page title
    st.title("Diabetes Prediction using Machine Learning(ML)")
    #st.image("E:\Pro_Stremlit بالعربي\image\diabetes_image.png")
    #st.image("image Multiple disease\diabetes_image.png")
    image_path = 'imageMultipledisease/diabetes_image.png'

# فتح الصورة وعرضها باستخدام Streamlit
    try:
        image = Image.open(image_path)
        st.image(image, caption='Diabetes Image', use_column_width=True)
    except Exception as e:
        st.error(f"Error opening image: {e}")
    
    s = 'Pregnancies 	Glucose 	BloodPressure 	SkinThickness 	Insulin 	BMI 	DiabetesPedigreeFunction 	Age'

    col1,col2,col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input("Pregnancies") 
    with col2:
        Glucose = st.text_input("Glucose")   
    with col3:
        BloodPressure = st.text_input("BloodPressure")
    with col1:
        SkinThickness = st.text_input("SkinThickness")
    with col2:
        Insulin = st.text_input("Insulin")
    with col3:
        BMI = st.text_input("BMI")
    with col1:
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction")
    with col2:
        Age = st.text_input("Age")
    
    df_diab = pd.DataFrame({'Pregnancies':Pregnancies,'Glucose':Glucose,'BloodPressure':BloodPressure,'SkinThickness':SkinThickness,
                            'Insulin':Insulin,'BMI':BMI,'DiabetesPedigreeFunction':DiabetesPedigreeFunction,'Age':Age},index=[0]) 
      
    # code for prediction    
    diab_prediction = ''
    
    #creating button of prediction
    pr = st.button("Diabetes Test Result")
    
    if pr :
        diab_pred = model_Diabetes.predict(df_diab)
        if (diab_pred[0] == 1):
            diab_prediction = 'The person is Diabetic'
            st.success(diab_prediction)
            #st.image("E:\Pro_Stremlit بالعربي\image\heart_diabetic.png",width=250)
            #st.image("image Multiple disease\heart_diabetic.png",width=250)
            image_path1 = r'imageMultipledisease/heart_diabetic.png'
            # فتح الصورة وعرضها باستخدام Streamlit
            try:
                image1 = Image.open(image_path1)
                st.image(image1, caption='Heart Disease',  use_column_width=True, width=250)
            except Exception as e:
                st.error(f"Error opening image: {e}")
                
            st.info(""" 
        Dear patient, don't worry! We are here to help you manage diabetes.
        You can maintain your health and successfully manage diabetes by following a healthy diet and exercising regularly. 
        Keep in touch with your healthcare team for the necessary guidance. 
        Together, we can keep you healthy and live your life to the fullest.
        """)
            
        else:
            diab_prediction = 'The Person is Not Diabetic'
            st.success(diab_prediction)
            #st.image("E:\Pro_Stremlit بالعربي\image\heart-strong.png",width=250)
            #st.image("image Multiple disease\heart-strong.png",width=250)
            image_path2 = 'imageMultipledisease/heart-strong.png' , 
            # فتح الصورة وعرضها باستخدام Streamlit
            try:
                image2 = Image.open(image_path2)
                st.image(image2, caption='Heart Strong',  use_column_width=True, width=250)
            except Exception as e:
                st.error(f"Error opening image: {e}")
                
            st.info("""
        Congratulations! You are not diagnosed with diabetes.
        To maintain your health and prevent this disease, we recommend following a balanced diet and exercising regularly.
        Keep a healthy lifestyle to stay in the best shape and health always.
        """)
            
#  0	118	84	47	230	45.8	0.551	31	1
# 10	139	80	0	0	27.1	1.441	57	0

# Heart Disease page
if (selected =='Heart Disease Prediction' ):
    st.title("Heart Disease Prediction using Machine Learning(ML)")
    #st.image("E:\Pro_Stremlit بالعربي\image\heart_Disease_image.png")
    #st.image("imageMultipledisease\heart_Disease_image.png")
    #image_path1 = 'imageMultipledisease\heart_Disease_image.png'
    image_path3 = 'imageMultipledisease/heart_Disease_image.png'

# فتح الصورة وعرضها باستخدام Streamlit
    try:
        image3 = Image.open(image_path3)
        st.image(image3, caption='Heart Image', use_column_width=True)
    except Exception as e:
        st.error(f"Error opening image: {e}")
    
    # H = Age 	Sex 	ChestPainType 	RestingBP 	Cholesterol 	FastingBS 	RestingECG 	MaxHR 	ExerciseAngina 	Oldpeak 	ST_Slope
    col1,col2,col3 = st.columns(3)
    with col1:
        Age = st.text_input("Age")
    with col2:
        Sex = st.text_input("Sex")
    with col3:
        ChestPainType = st.text_input("ChestPainType")
    with col1:
        RestingBP = st.text_input("RestingBP")
    with col2:
        Cholesterol = st.text_input("Cholesterol")        
    with col3:
        FastingBS = st.text_input("FastingBS > 120 mg/dl")
    with col1:
        RestingECG = st.text_input("RestingECG")
    with col2:
        MaxHR = st.text_input("MaxHR")
    with col3:
        ExerciseAngina = st.text_input("ExerciseAngina")      
    with col1:
        Oldpeak = st.text_input("Oldpeak ")
    with col2:
        ST_Slope = st.text_input("Slope of the peak exercise ST segment")
        
    df_hrt = pd.DataFrame({'Age':Age,'Sex':Sex,'ChestPainType':ChestPainType,
                           'RestingBP':RestingBP,'Cholesterol':Cholesterol,'FastingBS':FastingBS,
                           'RestingECG':RestingECG,'MaxHR':MaxHR,'ExerciseAngina':ExerciseAngina,
                           'Oldpeak':Oldpeak,'Slope of the peak exercise ST segment':ST_Slope},index=[0])  
    
    # code for prediction
    heart_pred = ''
    
    #creating button of prediction
    pr2 = st.button("Heart Test result")
    
    if pr2:
        hrt_pred = model_Heart.predict(df_hrt)
        if (hrt_pred[0]==1):
            heart_pred = 'The Person is Disease'
            st.success(heart_pred)
            #st.image("E:\Pro_Stremlit بالعربي\image\hrtdiseas.png.jpg",width=250)
            #st.image("image Multiple disease\hrtdiseas.png.jpg",width=250)
            image_path4 = r'imageMultipledisease/hrtdiseas.png.jpg'
            # فتح الصورة وعرضها باستخدام Streamlit
            try:
                image4 = Image.open(image_path4)
                st.image(image4, caption='Heart Disease' , use_column_width=True )
            except Exception as e:
                st.error(f"Error opening image: {e}")
                
            st.info( """
        Dear patient, don't be discouraged! We are here to support you in managing heart disease.
        Maintaining a healthy diet, exercising regularly, and following your doctor's advice can help you live a healthy and fulfilling life.
        Stay in regular contact with your healthcare provider for the best care and guidance.
        Together, we can ensure you live a happy and healthy life.
        """)
            
        else:
            heart_pred = 'The Person is Not Disease'
            st.success(heart_pred)
            #st.image("E:\Pro_Stremlit بالعربي\image\heart-strong.png",width=250)
            #st.image("image Multiple disease\heart-strong.png",width=250)
            image_path5 = r'imageMultipledisease/heart-strong.png'
            # فتح الصورة وعرضها باستخدام Streamlit
            try:
                image5 = Image.open(image_path5)
                st.image(image5, caption='Heart Strong', use_column_width=True , width=250)
            except Exception as e:
                st.error(f"Error opening image: {e}")
            
            st.info("""
        Great news! You are not diagnosed with heart disease.
        To keep your heart healthy, continue following a balanced diet, exercising regularly, and avoiding unhealthy habits.
        Maintain a heart-healthy lifestyle to ensure your well-being and vitality for years to come.
        """)
            
# 60	1	2	140	185	0	0	155	0	3	1	0	2	--> 0
# 48	1	1	130	245	0	0	180	0	0.2	1	0	2	--> 1




# Parkinsons Prediction Page
if (selected == 'Parkinsons Prediction'):
    st.title("Parkinsons Prediction using Machine Learning(ML)")
    #st.image("E:\Pro_Stremlit بالعربي\image\Parkinsons_hmage.jpg")
    #st.image("imageMultipledisease\Parkinsons_hmage.jpg")
    image_path6 = r'imageMultipledisease/Parkinsons_hmage.jpg'

# فتح الصورة وعرضها باستخدام Streamlit
    try:
        image6 = Image.open(image_path6)
        st.image(image6, caption='Parkinsons Image', use_column_width=False)
    except Exception as e:
        st.error(f"Error opening image: {e}")
    

    park_columns = [''' MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ,
        Jitter:DDP, MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA,
        NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE ''']


    col1,col2,col3,col4,col5 = st.columns(5)
    
    with col1:
        f1 =  st.text_input("MDVP Fo(Hz)") 
    with col2:
        f2 =  st.text_input("MDVP Fhi(Hz)")
    with col3:
        f3 =  st.text_input("MDVP Flo(Hz)")
    with col4:
        f4 =  st.text_input("MDVP jitter(%)") 
    with col5:
        f5 =  st.text_input("MDVP jitter(Abs)")
    with col1:
        f6 =  st.text_input("MDVP RAP")
    with col2:
        f7 =  st.text_input("MDVP PPQ") 
    with col3:
        f8 =  st.text_input("jitter DDP")
    with col4:
        f9 =  st.text_input("MDVP Shimmer")
    with col5:
        f10 = st.text_input("MDVP Shimmer(dB)") 
    with col1:
        f11 = st.text_input("Shimmer APQ3")
    with col2:
        f12 = st.text_input("Shimmer APQ5")
    with col3:
        f13 = st.text_input("MDVP APQ") 
    with col4:
        f14 = st.text_input("Shimmer DDA")
    with col5:
        f15 = st.text_input("NHR")
    with col1:
        f16 = st.text_input("HNR")
    with col2:
        f17 = st.text_input("RPDE")
    with col3:
        f18 = st.text_input("DFA")
    with col4:
        f19 = st.text_input("spread1")
    with col5:
        f20 = st.text_input("spread2")
    with col1:
        f21 = st.text_input("D2") 
    with col2:
        f22 = st.text_input("PPE")   
        
    df_prk = pd.DataFrame({'MDVP Fo(Hz)':f1,'MDVP Fhi(Hz)':f2,'MDVP Flo(Hz)':f3,
                           'MDVP jitter(%)':f4,'MDVP jitter(Abs)':f5,
                           'MDVP RAP':f6,'MDVP PPQ':f7,'jitter DDP':f8,
                           'MDVP Shimmer':f9,'MDVP Shimmer(dB)':f10,'Shimmer APQ3':f11,'Shimmer APQ5':f12,
                           'MDVP APQ':f13,'Shimmer DDA':f14,'NHR':f15,
                           'HNR':f16,'RPDE':f17,'DFA':f18,'spread1':f19,'spread2':f20,
                           'D2':f21,'PPE':f22,},index=[0])        
        
        
    #codel for prediction 
    park_pred = ''
    
    #creating button of prediction
    pr3 = st.button("Parkinson's Test Result")

    if pr3:
        prk_prd = model_Parkinsons.predict(df_prk)
        if (prk_prd[0] == 1):
            park_pred = "The person has Parkinson's disease"
            st.success(park_pred())
            st.info("""
        Dear patient, we understand that a diagnosis of Parkinson's disease can be challenging. 
        Remember, you are not alone—we are here to support you every step of the way. 
        By following the treatment plan recommended by your doctor, staying active, and maintaining a positive outlook, 
        you can manage the symptoms effectively and enjoy a good quality of life. 
        Don't hesitate to reach out to your healthcare team whenever you need guidance or support.
        """)
            

        else:
            park_pred = "The person dose not have Parkinson's disease"
            st.success(park_pred)
            st.info("""
        Wonderful news! You are not diagnosed with Parkinson's disease.
        To continue protecting your health and well-being, stay active, eat a balanced diet, and engage in activities that keep your mind sharp. 
        Continue with a healthy lifestyle to reduce the risk of neurological conditions in the future.
        """)
            
# 119.992	157.302	74.997	0.00784	0.00007	0.0037	0.00554	0.01109	0.04374	0.426	0.02182	0.0313	0.02971	0.06545	0.02211	21.033 0.414783	0.815285 -4.813031	0.266482	2.301442	0.284654	--> 1	
# 197.076	206.896	192.055	0.00289	0.00001	0.00166	0.00168	0.00498	0.01098	0.097	0.00563	0.0068	0.00802	0.01689	0.00339	26.775 0.422229	0.741367 -7.3483	0.177551	1.743867	0.085569	--> 0
# 120.267	137.244	114.82	0.00333	0.00003	0.00155	0.00202	0.00466	0.01608	0.14	0.00779	0.00937	0.01351	0.02337	0.00607	24.886	1	0.59604	0.764112	-5.634322	0.257682	1.854785	0.211756
	









