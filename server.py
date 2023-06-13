import numpy as np
import scipy.integrate
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/")
def hello():
    bet_adsorbed_ratio = 0.6
    dalton_evaporation_constant = 0.05
    control_rate = 100000
    dt = 10 / 3600

    target_H = 2000.
    target_L = 1000.
    thredhold_step = 32

    temperature = int(request.args.get("temp")) / 10.
    water_vapor_pressure = max(int(request.args.get("pres")) / 10., 0)
    infiltration_rate = max(int(request.args.get("infi")) / 10., 0.1)

    print(f'temperature = {temperature}')
    print(f'water_vapor_pressure = {water_vapor_pressure}')
    print(f'infiltration_rate = {infiltration_rate}')

    def dh_dt(h, _):
        saturated_water_vapor_pressure = 1.455 * temperature - 3.980
        bet_balance_pressure = (0.853 + 0.03 * np.log(bet_adsorbed_ratio * h + 1e-5) - 0.0002715 * (bet_adsorbed_ratio * h)) * saturated_water_vapor_pressure
        return - dalton_evaporation_constant * (bet_balance_pressure - water_vapor_pressure) - infiltration_rate * h if h > 0 else 0

    hum = scipy.integrate.odeint(dh_dt, 1.5 * target_H, np.arange(0, 6, dt))[:, 0]
    LT, DT = np.meshgrid(np.arange(0, 2048, thredhold_step), np.arange(0, 2048, thredhold_step))
    HT = LT + DT

    L_cost = np.sum(((LT < hum[:, None, None]) & (hum[:, None, None] < target_L)) * hum[:, None, None], axis=0) + 0.5 * (target_L - LT)**2 / (control_rate * dt) * (target_L > LT)
    H_cost = np.sum(((target_H < hum[:, None, None]) & (hum[:, None, None] < HT)) * hum[:, None, None], axis=0) + 0.5 * (HT - target_H)**2 / (control_rate * dt) * (HT > target_H)

    cost = L_cost + H_cost + (8000 / (np.maximum(np.searchsorted(-hum, -LT) - np.searchsorted(-hum, -HT), 0) + 1))**2
    td, tl = np.argwhere(cost == np.min(cost))[0] * thredhold_step
    th = td + tl

    print(f'thredhold = ({tl:.1f}, {th:.1f})')

    return f'{tl:d}, {th:d}'