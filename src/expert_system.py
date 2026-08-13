# expert_system.py

from config import CROP_RULES, DEFAULT_RULES

CROP_INFO = {
    "Rice": {"Scientific Name": "Oryza sativa", "Season": "Kharif", "Typical Yield": "3-6 tons/ha"},
    "Maize": {"Scientific Name": "Zea mays", "Season": "Kharif", "Typical Yield": "2-5 tons/ha"},
    "Cotton": {"Scientific Name": "Gossypium", "Season": "Kharif", "Typical Yield": "1-2 tons/ha"},
    "Mango": {"Scientific Name": "Mangifera indica", "Season": "Summer/Pre-Monsoon", "Typical Yield": "10-15 tons/ha"},
    "Apple": {"Scientific Name": "Malus domestica", "Season": "Winter/Spring", "Typical Yield": "20-30 tons/ha"},
    "Chickpea": {"Scientific Name": "Cicer arietinum", "Season": "Rabi", "Typical Yield": "1-2.5 tons/ha"},
    "Kidneybeans": {"Scientific Name": "Phaseolus vulgaris", "Season": "Kharif/Rabi", "Typical Yield": "1-2 tons/ha"},
    "Pigeonpeas": {"Scientific Name": "Cajanus cajan", "Season": "Kharif", "Typical Yield": "1.5-2.5 tons/ha"},
    "Mothbeans": {"Scientific Name": "Vigna aconitifolia", "Season": "Kharif", "Typical Yield": "0.5-1 tons/ha"},
    "Mungbean": {"Scientific Name": "Vigna radiata", "Season": "Zaid/Kharif", "Typical Yield": "1-1.5 tons/ha"},
    "Blackgram": {"Scientific Name": "Vigna mungo", "Season": "Kharif/Zaid", "Typical Yield": "1-2 tons/ha"},
    "Lentil": {"Scientific Name": "Lens culinaris", "Season": "Rabi", "Typical Yield": "1-1.5 tons/ha"},
    "Pomegranate": {"Scientific Name": "Punica granatum", "Season": "Year-round", "Typical Yield": "10-15 tons/ha"},
    "Banana": {"Scientific Name": "Musa", "Season": "Year-round", "Typical Yield": "30-50 tons/ha"},
    "Grapes": {"Scientific Name": "Vitis vinifera", "Season": "Winter/Summer", "Typical Yield": "20-30 tons/ha"},
    "Watermelon": {"Scientific Name": "Citrullus lanatus", "Season": "Summer", "Typical Yield": "20-30 tons/ha"},
    "Muskmelon": {"Scientific Name": "Cucumis melo", "Season": "Summer", "Typical Yield": "15-20 tons/ha"},
    "Orange": {"Scientific Name": "Citrus x sinensis", "Season": "Winter", "Typical Yield": "15-25 tons/ha"},
    "Papaya": {"Scientific Name": "Carica papaya", "Season": "Year-round", "Typical Yield": "30-50 tons/ha"},
    "Coconut": {"Scientific Name": "Cocos nucifera", "Season": "Year-round", "Typical Yield": "10000-15000 nuts/ha"},
    "Jute": {"Scientific Name": "Corchorus", "Season": "Kharif", "Typical Yield": "2-3 tons/ha"},
    "Coffee": {"Scientific Name": "Coffea", "Season": "Monsoon", "Typical Yield": "1-2 tons/ha"},
}

def expert_system_insights(crop_name, values):
    """
    Generate logic-based alerts, guidelines, fertilizer suggestions, and yield estimates based on soil and weather conditions.
    values: [N, P, K, temperature, humidity, ph, rainfall]
    """
    N, P, K, temperature, humidity, ph, rainfall = values
    alerts = []
    guidelines = []
    fertilizers = []

    # Handle capitalization (dataset might have lowercase like 'rice')
    crop_title = crop_name.title() if isinstance(crop_name, str) else crop_name
    rules = CROP_RULES.get(crop_title, DEFAULT_RULES)
    info = CROP_INFO.get(crop_title, {"Scientific Name": "Unknown", "Season": "Varies", "Typical Yield": "Varies"})
    
    nutrient_rules = rules.get("nutrients", DEFAULT_RULES["nutrients"])
    rainfall_min, rainfall_max = rules.get("rainfall", DEFAULT_RULES["rainfall"])
    ph_min, ph_max = rules.get("ph", DEFAULT_RULES["ph"])

    # Soil summary
    soil_status = "Optimal"
    
    # Nutrient checks and Fertilizer Suggestions
    observed_nutrients = {"N": N, "P": P, "K": K}
    
    if N < nutrient_rules["N"][0]:
        alerts.append(f"Nitrogen (N) is critically low for {crop_title}.")
        fertilizers.append("Apply Urea or Ammonium Nitrate to quickly boost Nitrogen levels.")
        soil_status = "Deficient in Nitrogen"
    elif N > nutrient_rules["N"][1]:
        alerts.append("Nitrogen is above the recommended range. Excess N can delay flowering.")
        
    if P < nutrient_rules["P"][0]:
        alerts.append(f"Phosphorus (P) is critically low for {crop_title}.")
        fertilizers.append("Apply Superphosphate or DAP (Diammonium Phosphate) for root development.")
        if soil_status == "Optimal": soil_status = "Deficient in Phosphorus"
        else: soil_status += " & Phosphorus"
    elif P > nutrient_rules["P"][1]:
        alerts.append("Phosphorus is above the recommended range. May cause micronutrient lock-up.")

    if K < nutrient_rules["K"][0]:
        alerts.append(f"Potassium (K) is critically low for {crop_title}.")
        fertilizers.append("Apply Muriate of Potash (MOP) to improve disease resistance and crop quality.")
        if soil_status == "Optimal": soil_status = "Deficient in Potassium"
        else: soil_status += " & Potassium"
    elif K > nutrient_rules["K"][1]:
        alerts.append("Potassium is above the recommended range. Can interfere with Magnesium uptake.")

    if not fertilizers:
        fertilizers.append("Soil nutrients are well-balanced. Use organic compost or light NPK mixtures to maintain fertility.")

    # Rainfall checks
    if rainfall < rainfall_min:
        alerts.append(f"Rainfall ({rainfall}mm) is lower than recommended. High risk of water stress.")
    elif rainfall > rainfall_max:
        alerts.append(f"Rainfall ({rainfall}mm) is higher than ideal. Risk of waterlogging and fungal diseases.")

    # pH checks
    if ph < ph_min:
        alerts.append(f"Soil is too acidic (pH {ph}). Nutrient lockout is possible.")
        fertilizers.append("Apply Agricultural Lime to raise soil pH to optimal levels.")
    elif ph > ph_max:
        alerts.append(f"Soil is too alkaline (pH {ph}).")
        fertilizers.append("Apply Elemental Sulfur or acidifying organic matter to lower soil pH.")

    # Temperature checks
    if temperature < 15:
        alerts.append(f"Temperature ({temperature}°C) is quite low. Growth may stunt.")
    elif temperature > 35:
        alerts.append(f"Temperature ({temperature}°C) is very high. High heat stress risk.")

    # Guidelines compilation
    guidelines.append(f"Crop-specific management: {rules.get('notes', DEFAULT_RULES['notes'])}")
    guidelines.append(f"Maintain soil pH between {ph_min} and {ph_max}.")
    guidelines.append("Monitor soil moisture weekly and adjust irrigation according to rainfall.")

    # Yield Estimation Heuristic
    if len(alerts) == 0:
        yield_estimate = "Excellent (Close to Max Potential)"
    elif len(alerts) <= 2:
        yield_estimate = "Good (Minor Limitations)"
    elif len(alerts) <= 4:
        yield_estimate = "Moderate (Needs Intervention)"
    else:
        yield_estimate = "Poor (Highly Unfavorable Conditions)"

    insights = {
        "alerts": alerts,
        "guidelines": guidelines,
        "fertilizers": fertilizers,
        "soil_status": soil_status,
        "yield_estimate": yield_estimate,
        "crop_info": info
    }

    return insights
