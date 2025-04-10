import pandas as pd
import numpy as np

# Lists for synthetic data
ethiopian_names = ["Meklit","Chala","Mohammed","Sofia","Kedir" "Tewodros", "Selamawit", "Dawit", "Frehiwot", "Yohannes", 
                   "Birtukan", "Samuel", "Eden", "Abel", "Kalkidan", "Hana", "Nahom", 
                   "Ephrem", "Rahel", "Zewditu", "Tigist", "Senait", "Mulat", "Tsehay", 
                   "Mulugeta", "Yemesrach", "Marta", "Kassa", "Bethel", "Yared", "Lidia", 
                   "Sifan", "Mihret", "Kiros", "Genet", "Seyoum", "Belay", "Girmay","Hagos","Driba", 
                   "Ruth", "Fikirte", "Temesgen", "Alem", "Abebe", "Atnafu", "Jemal", 
                   "Mekonnen", "Kebede", "Marta", "Kidane", "Zelalem", "Abiy", "Sewit"]

# Real Ethiopian Geography-related YouTube channels and websites (functional and relevant)


channels = [
 
"https://www.youtube.com/@DistributedSystems",
"https://www.youtube.com/@TheTechCave",
"https://www.youtube.com/@LearnersCoach",
"https://www.youtube.com/@education4uofficial",
"https://www.youtube.com/@6.824",
"https://www.youtube.com/@engineeringstudent7572",
"https://www.youtube.com/@HashiCorp",
"https://www.youtube.com/@distributedsystems7634",
"https://www.youtube.com/@Intellipaat",
"https://www.youtube.com/@freecodecamp",
"https://www.youtube.com/@CloudxLab",
"https://www.youtube.com/@makingITsimple1",
"https://www.geeksforgeeks.org/what-is-a-distributed-system/",
"https://www.tutorialspoint.com/Distributed-Systems",
"https://hazelcast.com/foundations/distributed-computing/distributed-computing/",
"https://arxiv.org/pdf/0911.4395",


]
# Learning strategies
strategies = [
    "Spaced Repetition", "Mind Mapping", "Practice Quizzes", "Peer Teaching",
    "Flashcards", "Video Summaries", "Case Studies", "Group Study", "Self-Testing"
]

# Generate 200 rows of data
np.random.seed(42)
data = []
for _ in range(200):
    name = np.random.choice(ethiopian_names) + " " + np.random.choice(["Abebe", "Getachew", "Tesfaye", "Kebede", "Assefa", "Lemma","Amare","weldehawaryat","Zinabu","Shimuye","Asrat", "Hailu", "Tadesse", "Girma", "Worku","Seid","Jenberu","Mekonnen","Fentaw"])
    grade = np.random.choice(["A", "A+"], p=[0.4, 0.6])
    channel = np.random.choice(channels)
    strategy = np.random.choice(strategies)
    study_hours = np.random.randint(5, 20)
    study_time = np.random.choice(["Morning", "Afternoon", "Evening", "Night"])
    notes = np.random.choice([
        "Used Anki flashcards", "Focused on documentaries", "Color-coded notes", 
        "Practiced past exams", "Joined online study groups", "Drew diagrams", 
        "Analyzed case studies", "Wrote summaries", "Timed simulations"
    ])
    
    data.append([
        name, grade, channel, strategy, study_hours, study_time, "English/Amharic", notes
    ])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "Name", "Grade", "Recommended Channel/Link", 
    "Learning Strategy", "Study Hours Per Week", "Preferred Study Time", 
    "Language of Instruction", "Additional Notes"
])

# Save as CSV (make sure to specify a path if needed)
df.to_csv("Introduction to Distributed Systems.csv", index=False)
print("CSV file generated!")