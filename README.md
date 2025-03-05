1. Generate config files:
   - each scenario needs 1 config for train, and num_test instances for test
2. Call env in policy file, load config file and run policy
3. Parameters
   - battery capacity: 62 kWh
   - energy consumption: 0.25 kWh/km (0.125 kWh per minute)
   - speed: 30 km/hour
   - charging power: 50kW (fast), 11 kW (slow)
   - fare: base fare 8$ + 3.1$/km (1.55$ per minute)
   - charging price: 0.58$/kwh (highest), 0.09 $/kwh(lowest)
   - travel cost: 0.53$/km = 0.265$ per minute
4. Convert to SoCs:
   - energy consumption per minute: 0.125/62=0.002
   - charging power per minute: 50/60/62=0.013
