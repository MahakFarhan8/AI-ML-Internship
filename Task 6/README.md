# 🏷️ Support Ticket Auto-Tagging with Groq LLM  

---

## 🎯 Objective  
Automatically tag customer support tickets into predefined categories using the **Groq API** and **Large Language Models (LLMs)**.  

---

## 📂 Dataset  
- **Source**: Customer support ticket dataset (`tickets.csv`)  
- **Key Columns**:  
  - `Ticket ID`  
  - `Ticket Type`  
  - `Ticket Subject`  
  - `Ticket Description` *(used for classification)*  
  - `Ticket Priority`  
  - `Ticket Status`  

---

## 📝 Instructions  
1. Load dataset (`tickets.csv`).  
2. Use **Groq LLM** for classification of tickets.  
3. Apply:  
   - **Zero-Shot Learning** → Classify without training, only prompt.  
   - **Few-Shot Learning** → Improve accuracy using few labeled examples.  
4. Output top **3 most probable tags per ticket**.  
5. Save results into `classified_tickets.csv`.  

---

## 🛠️ Skills Gained  
- ✅ Prompt Engineering  
- ✅ LLM-based Text Classification  
- ✅ Zero-Shot & Few-Shot Learning  
- ✅ Multi-Class Prediction and Ranking  
- ✅ Practical use of Groq API  

---

## 📌 Categories Used  
- Technical Issue  
- Billing Issue  
- Account Management  
- Product Setup  
- Warranty/Return  
- General Inquiry  

---

## 🚀 Output Example  

| Ticket Description | Zero-Shot Prediction | Few-Shot Prediction |
|--------------------|----------------------|----------------------|
| "I'm having an issue with my GoPro setup..." | `Technical Issue` | `Product Setup` |

---
