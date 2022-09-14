# **Prep Assist**

## **Motivation**

The primary users are job seekers. It will also be useful for users who want to find greater and better opportunities in the job market. People often struggle with preparing for interviews especially the non-technical interviews. While the candidates might be well-versed in having good technical knowledge, the area where they want more improvement is in the communication skills. On top of that, every candidate has their own special needs and areas where they need/want improvement. Keeping this in mind, the main technologies that can help are based on Machine Learning (ML) and Natural Language Processing (NLP).

> A natural language based intelligent assistant for helping preparation for interview preparation. Aided with ability to conduct human-like conversation, and understanding the candidate's behavioral characteristics and communication skills, the application can help a candidate to prepare better for interviews by conducting personalized mock interviews, giving feedbacks and give reports on the candidate's performance.

<br>

## **Design**

The core NLP related tasks that application performs are - Question-Answering, Converstation/Dialogue System, Sentiment Analysis, and Behavior Analysis. W also need to provide personalized recommendations to the user which is based on ML, and reports of performance in an understandable way using the various Data Analysis techniques.

We are mainly focused on the verbal communication details, and hence the interface to interact with the assistant will be text based. A natural dialogue-like setting will be implemented where the user and the assistant can communicate with a series to text. Based on the text input by the user, the assistant would generate follow up questions, keeping the following things in perspective - the on-going topic, sentiment/behavior of the user, and the complexity of the interview session. It also needs to adapt the difficulty of the interview based on sentiments like nervousness, stress and emotions which a user feels which being in an interview setup. The dialogue system needs to be trained on dataset for interviews. It will also require a dataset of possible questions and answers to learn.

<br>

## **Analysis**

The proposed application uses expertise in multiple tasks to provide a coherent and robust assisting system to the user. However, currently there are very limited to no datasets which are particularly focused on interview dialogues, which is the biggest hurdle in implementing the core functionality. Moreover, the application requires knowledge from experts in the field of behavioral sciences, interviewing as well as recruiters from different fields. Currently it is only based on analyzing textual responses of a candidate and does not account for the visual cues which can be added as an improvement in the next iteration.

<br>

**Impact**

Having such an application would not only boost confidence, but it will also reduce anxiety and fatigue related with job interview preparation to a great extent. It reduces the uncertainty about overall presentation. Moreover, it can also help recruiter have a well prepared candidate and also reduce their turn-around time while finding a new hire, leading to a better recruitement experience.
