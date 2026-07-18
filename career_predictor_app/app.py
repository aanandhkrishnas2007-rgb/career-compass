import os
import pandas as pd
import joblib
from flask import Flask, render_template, request
from graphviz import Digraph

app = Flask(__name__)

# ---------------------------------------------------------------
# Load trained artifacts once, at startup (not per-request)
# ---------------------------------------------------------------
rf = joblib.load("model/rf_model.pkl")
le = joblib.load("model/label_encoder.pkl")
model_columns = joblib.load("model/model_columns.pkl")
category_options = joblib.load("model/category_options.pkl")

ROADMAPS = {
    "Web Developer": ["HTML", "CSS", "JavaScript", "Bootstrap", "React.js",
                       "Node.js", "Express.js", "MongoDB / MySQL", "Git & GitHub",
                       "Build Full Stack Projects", "Internship", "Web Developer"],
    "Data Scientist": ["Python", "Statistics", "Mathematics", "SQL",
                        "Pandas & NumPy", "Data Visualization", "Machine Learning",
                        "Deep Learning", "Build ML Projects", "Kaggle",
                        "Internship", "Data Scientist"],
    "Database Administrator": ["Database Fundamentals", "SQL", "MySQL",
                                "PostgreSQL", "Oracle Database", "Database Security",
                                "Backup & Recovery", "Performance Tuning",
                                "Cloud Databases", "Internship", "Database Administrator"],
    "Data Analyst": ["Excel", "SQL", "Python", "Pandas", "Statistics", "Power BI",
                      "Tableau", "Data Cleaning", "Dashboard Projects",
                      "Internship", "Data Analyst"],
    "UI/UX Designer": ["Design Principles", "Color Theory", "Typography",
                        "Wireframing", "Figma", "Adobe XD", "Prototyping",
                        "User Research", "Portfolio", "Internship", "UI/UX Designer"],
    "Cybersecurity Specialist": ["Networking", "Linux", "Python",
                                  "Operating Systems", "Cryptography", "Ethical Hacking",
                                  "Penetration Testing", "Security Tools",
                                  "Security Certifications", "Internship",
                                  "Cybersecurity Specialist"],
    "Business Analyst": ["Business Fundamentals", "Excel", "SQL", "Power BI",
                          "Tableau", "Data Analysis", "Requirement Gathering",
                          "Agile & Scrum", "Communication Skills", "Internship",
                          "Business Analyst"],
    "Machine Learning Engineer": ["Python", "Statistics", "Linear Algebra",
                                   "Machine Learning", "Scikit-learn", "Deep Learning",
                                   "TensorFlow/PyTorch", "MLOps", "Deploy Models",
                                   "Internship", "Machine Learning Engineer"],
    "Software Developer": ["Programming Fundamentals", "Java / Python / C++",
                            "Data Structures", "Algorithms",
                            "Object-Oriented Programming", "Git & GitHub", "Database",
                            "System Design Basics", "Projects", "Internship",
                            "Software Developer"],
    "Cloud Engineer": ["Linux", "Networking", "Python", "AWS / Azure / GCP",
                        "Virtualization", "Docker", "Kubernetes", "Terraform",
                        "CI/CD", "Internship", "Cloud Engineer"],
}

SKILL_FIELDS = [
    ("Programming_Skill", "Programming"),
    ("Python_Skill", "Python"),
    ("Java_Skill", "Java"),
    ("SQL_Skill", "SQL"),
    ("Machine_Learning_Skill", "Machine Learning"),
    ("Web_Development_Skill", "Web Development"),
    ("Networking_Skill", "Networking"),
    ("Communication_Skill", "Communication"),
    ("Problem_Solving_Skill", "Problem Solving"),
    ("Creativity_Skill", "Creativity"),
]


@app.route("/")
def home():
    return render_template("index.html", category_options=category_options,
                            skill_fields=SKILL_FIELDS)


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    user_data = {
        "Age": int(form["Age"]),
        "Degree": form["Degree"],
        "CGPA": float(form["CGPA"]),
        "Interest_Area": form["Interest_Area"],
        "Internship": form["Internship"],
        "Number_of_Projects": int(form["Number_of_Projects"]),
        "Certifications": int(form["Certifications"]),
    }
    for field, _ in SKILL_FIELDS:
        user_data[field] = int(form[field])

    user_df = pd.DataFrame([user_data])
    user_df = pd.get_dummies(user_df)
    user_df = user_df.reindex(columns=model_columns, fill_value=0)

    prediction = rf.predict(user_df)
    predicted_career = le.inverse_transform(prediction)[0]

    probabilities = rf.predict_proba(user_df)[0]
    result = pd.DataFrame({
        "Career": le.classes_,
        "Suitability": probabilities * 100
    }).sort_values(by="Suitability", ascending=False)

    top5 = result.head(5).to_dict("records")

    roadmap_image = generate_roadmap(predicted_career)

    return render_template(
        "result.html",
        predicted_career=predicted_career,
        top5=top5,
        roadmap_image=roadmap_image,
    )


def generate_roadmap(career):
    """Render the roadmap to a PNG in static/roadmaps and return its filename."""
    steps = ROADMAPS.get(career)
    if not steps:
        return None

    safe_name = career.replace(" ", "_").replace("/", "_")
    out_path = os.path.join("static", "roadmaps", safe_name)  # graphviz appends .png

    dot = Digraph(comment="Career Roadmap", format="png")
    dot.attr(rankdir="TB")
    for i, step in enumerate(steps):
        dot.node(str(i), step, shape="box")
    for i in range(len(steps) - 1):
        dot.edge(str(i), str(i + 1))

    dot.render(out_path, cleanup=True)
    return f"roadmaps/{safe_name}.png"


if __name__ == "__main__":
    app.run(debug=True)
