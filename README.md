# Research-on-Android-Malware-Detection-Based-on-Description-Permission-Fidelity
The mobile Internet has penetrated into every corner of human social life because of its high portability advantages. The Android system provides a broad environment to developers which makes Android applications both in the number and coverage of the field are growing rapidly. To bring convenience to the user at the same time also make its security issues become the focus. The application descriptions on app store are a means for the developers to communicate the application functionality to the users. From the security and privacy standpoint, these descriptions should thus indicate the reasons for the permissions requested by an application, either explicitly or implicitly. We call it fidelity of descriptions to permissions. This article argues that the Android application that doesn't meet the fidelity is malware. From this point of view, This article transforms the detection of malicious Android applications into a new problem.This paper proposes a text classification method based on Deep Learning for description-permission fidelity. This method can effectively classify each descriptive statement of an Android application by permission to determine whether an application's description can express the declared permission. Through this way can help users to further identify malicious applications. Meanwhile, you can feed the developer back about the shortcomings of the description and help to enhance the overall credibility of the application store. In this paper, the model is implemented under the Keras architecture and experimented on an open data set in order to verify the validity of the proposed method. The result shows that the effect is better than the previous method with precision of 98.26%, recall of 98.27%, F-score of 98.21% and accuracy of 98.24% as opposed to previous 82.2%,81.5%,82.8 and 97.3% respectively. In addition to helping users detect malware, it can also be used to feedback to the developers on the quality of descriptions, and help to enhance the trustworthiness  of the app store.