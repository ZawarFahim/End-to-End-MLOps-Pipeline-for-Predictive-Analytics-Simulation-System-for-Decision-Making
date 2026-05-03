/**
 * Global agricultural locations dataset — 120 cities worldwide.
 * Each entry: { name, lat, lng, temp (°C avg), rain (mm/yr), humidity (%), N, P, K }
 *
 * Climate data is approximate annual averages used for ML feature input.
 * Covers Asia, Africa, Europe, Americas, Oceania.
 */

const globalLocations = [
  // ── SOUTH ASIA ───────────────────────────────────────────────────────────
  { name: "Lahore", lat: 31.52, lng: 74.36, temp: 25, rain: 620, humidity: 65, N: 88, P: 45, K: 42 },
  { name: "Karachi", lat: 24.86, lng: 67.01, temp: 29, rain: 180, humidity: 72, N: 65, P: 30, K: 36 },
  { name: "Islamabad", lat: 33.68, lng: 73.05, temp: 21, rain: 1150, humidity: 60, N: 74, P: 40, K: 39 },
  { name: "Multan", lat: 30.16, lng: 71.52, temp: 28, rain: 185, humidity: 58, N: 84, P: 43, K: 41 },
  { name: "Peshawar", lat: 34.02, lng: 71.52, temp: 24, rain: 400, humidity: 55, N: 76, P: 38, K: 35 },
  { name: "Quetta", lat: 30.18, lng: 66.98, temp: 20, rain: 260, humidity: 45, N: 64, P: 29, K: 30 },
  { name: "Delhi", lat: 28.61, lng: 77.21, temp: 30, rain: 800, humidity: 60, N: 90, P: 48, K: 44 },
  { name: "Mumbai", lat: 19.08, lng: 72.88, temp: 28, rain: 2200, humidity: 75, N: 85, P: 50, K: 46 },
  { name: "Kolkata", lat: 22.57, lng: 88.36, temp: 27, rain: 1800, humidity: 78, N: 80, P: 42, K: 40 },
  { name: "Chennai", lat: 13.08, lng: 80.27, temp: 29, rain: 1400, humidity: 74, N: 78, P: 40, K: 38 },
  { name: "Bangalore", lat: 12.97, lng: 77.59, temp: 24, rain: 970, humidity: 65, N: 75, P: 38, K: 36 },
  { name: "Hyderabad", lat: 17.38, lng: 78.49, temp: 27, rain: 900, humidity: 60, N: 76, P: 39, K: 37 },
  { name: "Dhaka", lat: 23.81, lng: 90.41, temp: 27, rain: 2050, humidity: 80, N: 82, P: 44, K: 42 },
  { name: "Colombo", lat: 6.93, lng: 79.85, temp: 28, rain: 2400, humidity: 78, N: 84, P: 46, K: 43 },
  { name: "Kathmandu", lat: 27.70, lng: 85.32, temp: 18, rain: 1400, humidity: 70, N: 72, P: 36, K: 34 },

  // ── EAST & SOUTHEAST ASIA ────────────────────────────────────────────────
  { name: "Beijing", lat: 39.91, lng: 116.39, temp: 13, rain: 600, humidity: 55, N: 76, P: 38, K: 36 },
  { name: "Shanghai", lat: 31.23, lng: 121.47, temp: 18, rain: 1150, humidity: 72, N: 80, P: 42, K: 40 },
  { name: "Guangzhou", lat: 23.13, lng: 113.26, temp: 23, rain: 1700, humidity: 78, N: 85, P: 46, K: 44 },
  { name: "Chengdu", lat: 30.57, lng: 104.07, temp: 17, rain: 900, humidity: 80, N: 78, P: 40, K: 38 },
  { name: "Tokyo", lat: 35.69, lng: 139.69, temp: 16, rain: 1530, humidity: 68, N: 74, P: 38, K: 36 },
  { name: "Osaka", lat: 34.69, lng: 135.50, temp: 17, rain: 1300, humidity: 65, N: 72, P: 36, K: 34 },
  { name: "Seoul", lat: 37.57, lng: 126.98, temp: 13, rain: 1370, humidity: 62, N: 73, P: 37, K: 35 },
  { name: "Bangkok", lat: 13.75, lng: 100.52, temp: 29, rain: 1400, humidity: 80, N: 88, P: 48, K: 46 },
  { name: "Ho Chi Minh City", lat: 10.82, lng: 106.63, temp: 28, rain: 1800, humidity: 82, N: 86, P: 46, K: 44 },
  { name: "Hanoi", lat: 21.03, lng: 105.85, temp: 24, rain: 1680, humidity: 79, N: 84, P: 44, K: 42 },
  { name: "Manila", lat: 14.60, lng: 120.98, temp: 28, rain: 2000, humidity: 78, N: 85, P: 45, K: 43 },
  { name: "Kuala Lumpur", lat: 3.14, lng: 101.69, temp: 28, rain: 2630, humidity: 82, N: 86, P: 48, K: 46 },
  { name: "Singapore", lat: 1.35, lng: 103.82, temp: 27, rain: 2340, humidity: 84, N: 82, P: 44, K: 42 },
  { name: "Jakarta", lat: -6.21, lng: 106.85, temp: 28, rain: 1800, humidity: 80, N: 88, P: 50, K: 48 },
  { name: "Yangon", lat: 16.87, lng: 96.19, temp: 27, rain: 2700, humidity: 82, N: 83, P: 45, K: 43 },

  // ── CENTRAL ASIA & MIDDLE EAST ───────────────────────────────────────────
  { name: "Kabul", lat: 34.53, lng: 69.17, temp: 13, rain: 330, humidity: 40, N: 60, P: 28, K: 28 },
  { name: "Tehran", lat: 35.69, lng: 51.39, temp: 17, rain: 230, humidity: 38, N: 58, P: 26, K: 26 },
  { name: "Baghdad", lat: 33.34, lng: 44.40, temp: 25, rain: 150, humidity: 42, N: 62, P: 28, K: 27 },
  { name: "Riyadh", lat: 24.69, lng: 46.72, temp: 28, rain: 100, humidity: 30, N: 55, P: 22, K: 20 },
  { name: "Dubai", lat: 25.20, lng: 55.27, temp: 28, rain: 75, humidity: 60, N: 52, P: 20, K: 18 },
  { name: "Istanbul", lat: 41.01, lng: 28.98, temp: 15, rain: 820, humidity: 72, N: 76, P: 40, K: 38 },
  { name: "Tashkent", lat: 41.30, lng: 69.24, temp: 14, rain: 400, humidity: 55, N: 70, P: 34, K: 32 },

  // ── AFRICA ───────────────────────────────────────────────────────────────
  { name: "Cairo", lat: 30.04, lng: 31.24, temp: 22, rain: 25, humidity: 52, N: 50, P: 20, K: 18 },
  { name: "Alexandria", lat: 31.20, lng: 29.92, temp: 21, rain: 180, humidity: 68, N: 55, P: 25, K: 23 },
  { name: "Lagos", lat: 6.52, lng: 3.38, temp: 28, rain: 1630, humidity: 82, N: 85, P: 46, K: 44 },
  { name: "Abuja", lat: 9.07, lng: 7.40, temp: 27, rain: 1300, humidity: 74, N: 80, P: 42, K: 40 },
  { name: "Nairobi", lat: -1.29, lng: 36.82, temp: 18, rain: 860, humidity: 65, N: 74, P: 38, K: 36 },
  { name: "Addis Ababa", lat: 9.03, lng: 38.74, temp: 16, rain: 1180, humidity: 60, N: 72, P: 36, K: 34 },
  { name: "Dar es Salaam", lat: -6.79, lng: 39.21, temp: 27, rain: 1200, humidity: 78, N: 80, P: 42, K: 40 },
  { name: "Casablanca", lat: 33.57, lng: -7.59, temp: 18, rain: 430, humidity: 70, N: 68, P: 34, K: 32 },
  { name: "Tunis", lat: 36.82, lng: 10.17, temp: 19, rain: 470, humidity: 65, N: 68, P: 33, K: 31 },
  { name: "Johannesburg", lat: -26.20, lng: 28.04, temp: 15, rain: 760, humidity: 55, N: 70, P: 36, K: 34 },
  { name: "Cape Town", lat: -33.93, lng: 18.42, temp: 17, rain: 515, humidity: 72, N: 72, P: 38, K: 36 },
  { name: "Accra", lat: 5.56, lng: -0.20, temp: 27, rain: 740, humidity: 78, N: 76, P: 40, K: 38 },
  { name: "Khartoum", lat: 15.50, lng: 32.56, temp: 30, rain: 160, humidity: 34, N: 52, P: 22, K: 20 },
  { name: "Kampala", lat: 0.32, lng: 32.58, temp: 22, rain: 1400, humidity: 76, N: 82, P: 44, K: 42 },
  { name: "Kinshasa", lat: -4.33, lng: 15.32, temp: 25, rain: 1400, humidity: 80, N: 84, P: 46, K: 44 },
  { name: "Dakar", lat: 14.72, lng: -17.47, temp: 24, rain: 580, humidity: 74, N: 68, P: 34, K: 32 },

  // ── EUROPE ───────────────────────────────────────────────────────────────
  { name: "London", lat: 51.51, lng: -0.13, temp: 12, rain: 600, humidity: 76, N: 68, P: 34, K: 32 },
  { name: "Paris", lat: 48.86, lng: 2.35, temp: 12, rain: 650, humidity: 74, N: 70, P: 36, K: 34 },
  { name: "Berlin", lat: 52.52, lng: 13.40, temp: 10, rain: 590, humidity: 75, N: 66, P: 32, K: 30 },
  { name: "Madrid", lat: 40.42, lng: -3.70, temp: 15, rain: 450, humidity: 55, N: 64, P: 30, K: 28 },
  { name: "Rome", lat: 41.90, lng: 12.50, temp: 16, rain: 700, humidity: 70, N: 70, P: 35, K: 33 },
  { name: "Athens", lat: 37.98, lng: 23.73, temp: 19, rain: 400, humidity: 62, N: 66, P: 32, K: 30 },
  { name: "Amsterdam", lat: 52.37, lng: 4.90, temp: 10, rain: 840, humidity: 80, N: 68, P: 34, K: 32 },
  { name: "Warsaw", lat: 52.23, lng: 21.01, temp: 9, rain: 550, humidity: 74, N: 64, P: 30, K: 28 },
  { name: "Vienna", lat: 48.21, lng: 16.37, temp: 11, rain: 660, humidity: 72, N: 68, P: 34, K: 32 },
  { name: "Stockholm", lat: 59.33, lng: 18.07, temp: 7, rain: 540, humidity: 78, N: 62, P: 28, K: 26 },
  { name: "Bucharest", lat: 44.43, lng: 26.10, temp: 12, rain: 600, humidity: 70, N: 70, P: 36, K: 34 },
  { name: "Kiev", lat: 50.45, lng: 30.52, temp: 9, rain: 640, humidity: 74, N: 72, P: 36, K: 34 },
  { name: "Moscow", lat: 55.75, lng: 37.62, temp: 5, rain: 700, humidity: 76, N: 64, P: 30, K: 28 },
  { name: "Budapest", lat: 47.50, lng: 19.04, temp: 12, rain: 600, humidity: 70, N: 70, P: 35, K: 33 },
  { name: "Lisbon", lat: 38.72, lng: -9.14, temp: 17, rain: 770, humidity: 72, N: 68, P: 34, K: 32 },

  // ── NORTH AMERICA ────────────────────────────────────────────────────────
  { name: "New York", lat: 40.71, lng: -74.01, temp: 14, rain: 1170, humidity: 66, N: 74, P: 38, K: 36 },
  { name: "Los Angeles", lat: 34.05, lng: -118.24, temp: 19, rain: 380, humidity: 60, N: 66, P: 32, K: 30 },
  { name: "Chicago", lat: 41.88, lng: -87.63, temp: 10, rain: 940, humidity: 72, N: 76, P: 40, K: 38 },
  { name: "Houston", lat: 29.76, lng: -95.37, temp: 21, rain: 1290, humidity: 74, N: 80, P: 44, K: 42 },
  { name: "Phoenix", lat: 33.45, lng: -112.07, temp: 25, rain: 200, humidity: 28, N: 55, P: 24, K: 22 },
  { name: "Dallas", lat: 32.78, lng: -96.80, temp: 19, rain: 940, humidity: 64, N: 76, P: 40, K: 38 },
  { name: "Seattle", lat: 47.61, lng: -122.33, temp: 12, rain: 940, humidity: 76, N: 72, P: 38, K: 36 },
  { name: "Miami", lat: 25.76, lng: -80.19, temp: 24, rain: 1520, humidity: 78, N: 82, P: 46, K: 44 },
  { name: "Toronto", lat: 43.65, lng: -79.38, temp: 9, rain: 830, humidity: 68, N: 70, P: 36, K: 34 },
  { name: "Vancouver", lat: 49.25, lng: -123.12, temp: 11, rain: 1150, humidity: 78, N: 72, P: 38, K: 36 },
  { name: "Mexico City", lat: 19.43, lng: -99.13, temp: 16, rain: 700, humidity: 62, N: 76, P: 40, K: 38 },
  { name: "Guadalajara", lat: 20.66, lng: -103.35, temp: 20, rain: 980, humidity: 60, N: 78, P: 42, K: 40 },
  { name: "Havana", lat: 23.14, lng: -82.36, temp: 25, rain: 1300, humidity: 78, N: 80, P: 44, K: 42 },

  // ── CENTRAL & SOUTH AMERICA ──────────────────────────────────────────────
  { name: "Bogota", lat: 4.71, lng: -74.07, temp: 14, rain: 1010, humidity: 76, N: 76, P: 40, K: 38 },
  { name: "Lima", lat: -12.05, lng: -77.04, temp: 19, rain: 15, humidity: 80, N: 55, P: 24, K: 22 },
  { name: "Quito", lat: -0.23, lng: -78.52, temp: 13, rain: 1100, humidity: 72, N: 74, P: 38, K: 36 },
  { name: "Caracas", lat: 10.49, lng: -66.88, temp: 22, rain: 850, humidity: 72, N: 76, P: 40, K: 38 },
  { name: "São Paulo", lat: -23.55, lng: -46.63, temp: 19, rain: 1450, humidity: 76, N: 84, P: 48, K: 46 },
  { name: "Rio de Janeiro", lat: -22.91, lng: -43.17, temp: 24, rain: 1200, humidity: 80, N: 82, P: 46, K: 44 },
  { name: "Buenos Aires", lat: -34.61, lng: -58.38, temp: 17, rain: 1220, humidity: 70, N: 80, P: 44, K: 42 },
  { name: "Santiago", lat: -33.46, lng: -70.65, temp: 14, rain: 310, humidity: 60, N: 68, P: 34, K: 32 },
  { name: "La Paz", lat: -16.50, lng: -68.15, temp: 8, rain: 560, humidity: 58, N: 62, P: 28, K: 26 },
  { name: "Montevideo", lat: -34.91, lng: -56.17, temp: 16, rain: 1100, humidity: 74, N: 76, P: 40, K: 38 },
  { name: "Manaus", lat: -3.12, lng: -60.02, temp: 27, rain: 2300, humidity: 85, N: 88, P: 52, K: 50 },
  { name: "Brasilia", lat: -15.78, lng: -47.93, temp: 21, rain: 1550, humidity: 72, N: 82, P: 46, K: 44 },
  { name: "Asuncion", lat: -25.29, lng: -57.65, temp: 24, rain: 1300, humidity: 70, N: 78, P: 42, K: 40 },

  // ── OCEANIA ───────────────────────────────────────────────────────────────
  { name: "Sydney", lat: -33.87, lng: 151.21, temp: 18, rain: 1200, humidity: 67, N: 76, P: 40, K: 38 },
  { name: "Melbourne", lat: -37.81, lng: 144.96, temp: 15, rain: 650, humidity: 65, N: 72, P: 36, K: 34 },
  { name: "Brisbane", lat: -27.47, lng: 153.03, temp: 21, rain: 1150, humidity: 68, N: 78, P: 42, K: 40 },
  { name: "Perth", lat: -31.95, lng: 115.86, temp: 19, rain: 740, humidity: 57, N: 70, P: 34, K: 32 },
  { name: "Adelaide", lat: -34.93, lng: 138.60, temp: 17, rain: 550, humidity: 55, N: 68, P: 33, K: 31 },
  { name: "Auckland", lat: -36.86, lng: 174.77, temp: 15, rain: 1240, humidity: 74, N: 74, P: 38, K: 36 },
  { name: "Wellington", lat: -41.29, lng: 174.78, temp: 13, rain: 1290, humidity: 78, N: 72, P: 36, K: 34 },
  { name: "Port Moresby", lat: -9.44, lng: 147.18, temp: 27, rain: 1100, humidity: 72, N: 78, P: 42, K: 40 },
  { name: "Suva", lat: -18.14, lng: 178.44, temp: 25, rain: 3000, humidity: 82, N: 86, P: 48, K: 46 },
];

export default globalLocations;
