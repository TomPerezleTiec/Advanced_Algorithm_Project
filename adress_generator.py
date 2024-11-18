import random
import time
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

def get_random_addresses_in_barcelona(n):
    # Initialiser le géocodeur Nominatim de Geopy avec un user_agent unique
    geolocator = Nominatim(user_agent="adresse_generator_app_v1")

    # Coordonnées de la bounding box de Barcelone
    lat_min, lat_max = 41.291, 41.443
    lon_min, lon_max = 2.070, 2.230

    addresses = []
    attempts = 0  # Compteur pour limiter le nombre de tentatives

    while len(addresses) < n and attempts < 300:  # Limiter à 100 tentatives au maximum
        # Sélectionner des coordonnées aléatoires dans la bounding box de Barcelone
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)

        try:
            # Faire le géocodage inverse pour obtenir l'adresse avec un délai d'attente plus long
            location = geolocator.reverse((lat, lon), language="fr", exactly_one=True, addressdetails=True, timeout=10)

            if location:
                address_parts = location.raw.get('address', {})

                street = address_parts.get('road', '')  # Le nom de la rue
                number = address_parts.get('house_number', '')  # Le numéro de rue
                city = address_parts.get('city', '')  # La ville
                postcode = address_parts.get('postcode', '')  # Le code postal

                # Vérifier si le nom de la rue et le numéro de rue sont non vides
                if street and number:  # Les deux doivent être non vides
                    # Formater l'adresse de manière lisible avec un numéro de rue, le nom de la rue, et la ville
                    formatted_address = f"{street}, {number}, {postcode} Barcelona, Spain"
                    addresses.append(formatted_address)

        except GeocoderTimedOut:
            print(f"Erreur de délai d'attente pour les coordonnées ({lat}, {lon}), réessayer...")

        except Exception as e:
            print(f"Erreur lors du géocodage pour les coordonnées ({lat}, {lon}) : {e}")

        # Ajouter un délai de 2 secondes entre chaque requête pour éviter les limitations de Nominatim
        time.sleep(1)

        # Incrémenter le compteur de tentatives
        attempts += 1

    # Si nous n'avons pas atteint n adresses après 100 tentatives, avertir l'utilisateur
    if len(addresses) < n:
        print(f"Attention : Moins de {n} adresses valides trouvées après {attempts} tentatives.")

    return addresses

def generate_random_delivery_windows(n):
    windows = []
    # Fenêtre fixe pour le dépôt
    windows.append(("08:00", "18:00"))

    for _ in range(1, n):  # Démarrer à 1 puisque la première adresse est pour le dépôt
        # Générer une heure d'ouverture aléatoire entre 08:00 et 15:30 (pour une fermeture avant 17:30)
        opening_hour = random.randint(8, 15)
        opening_minute = random.choice([0, 15, 30, 45])

        # Calculer l'heure de fermeture avec une durée minimale de 2 heures
        closing_hour = opening_hour + 3
        closing_minute = random.choice([0, 15, 30, 45])

        # Vérifier que la fermeture ne dépasse pas 17:30
        if closing_hour > 17 or (closing_hour == 17 and closing_minute > 30):
            closing_hour = 17
            closing_minute = 30

        # Formatage des heures en chaîne de caractères
        opening_time = f"{opening_hour:02}:{opening_minute:02}"
        closing_time = f"{closing_hour:02}:{closing_minute:02}"

        # Ajouter la fenêtre de livraison à la liste
        windows.append((opening_time, closing_time))

    return windows

def create_excel_with_addresses(n):
    # Générer des adresses et des fenêtres de livraison
    random_addresses = get_random_addresses_in_barcelona(n)
    delivery_windows = generate_random_delivery_windows(n)

    # Créer un DataFrame avec les adresses et fenêtres de livraison
    data = {
        "Adresse": random_addresses,
        "Ouverture": [window[0] for window in delivery_windows],
        "Fermeture": [window[1] for window in delivery_windows]
    }

    # Convertir en DataFrame et enregistrer dans un fichier Excel
    df = pd.DataFrame(data)
    df.to_excel("clients_depot_barcelone.xlsx", index=False)
    print("Les données ont été enregistrées dans le fichier 'clients_depot_barcelone.xlsx'.")

create_excel_with_addresses(20)