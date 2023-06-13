#include <Arduino.h>
#include <Adafruit_AHTX0.h>
#include <HttpClient.h>
#include <WiFi.h>
#include <inttypes.h>
#include <stdio.h>
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs.h"
#include "nvs_flash.h"
#include <Fastor/Fastor.h>
#include <Servo.h>

#define servoPin 32
#define soilHumidityPin 36

constexpr double bet_adsorbed_ratio = 0.6;
constexpr double dalton_evaporation_constant = 0.05;
constexpr double control_rate = 100000;
constexpr double dt = 2 / 3600.;

double low_thredhold = 100;
double high_thredhold = 1500;

char ssid[50];
char pass[50];

using namespace Fastor;

void nvs_access()
{
  // Initialize NVS
  esp_err_t err = nvs_flash_init();
  if (err == ESP_ERR_NVS_NO_FREE_PAGES ||
      err == ESP_ERR_NVS_NEW_VERSION_FOUND)
  {
    // NVS partition was truncated and needs to be erased
    // Retry nvs_flash_init
    ESP_ERROR_CHECK(nvs_flash_erase());
    err = nvs_flash_init();
  }
  ESP_ERROR_CHECK(err);
  // Open
  Serial.printf("\n");
  Serial.printf("Opening Non-Volatile Storage (NVS) handle... ");
  nvs_handle_t my_handle;
  err = nvs_open("storage", NVS_READWRITE, &my_handle);
  if (err != ESP_OK)
  {
    Serial.printf("Error (%s) opening NVS handle!\n", esp_err_to_name(err));
  }
  else
  {
    Serial.printf("Done\n");
    Serial.printf("Retrieving SSID/PASSWD\n");
    size_t ssid_len = 50;
    size_t pass_len = 50;
    err = nvs_get_str(my_handle, "ssid", ssid, &ssid_len);
    err |= nvs_get_str(my_handle, "pass", pass, &pass_len);
    switch (err)
    {
    case ESP_OK:
      Serial.printf("Done\n");
      break;
    case ESP_ERR_NVS_NOT_FOUND:
      Serial.printf("The value is not initialized yet!\n");
      break;
    default:
      Serial.printf("Error (%s) reading!\n", esp_err_to_name(err));
    }
  }
  // Close
  nvs_close(my_handle);
}

Servo actor;
Adafruit_AHTX0 aht;
Adafruit_Sensor *aht_humidity, *aht_temp;

Tensor<double, 4> x; // soil moisture, water vapor pressure (hPa), air temperature (°C), permeability
Tensor<double, 4, 4> Q, P, I;
Tensor<double, 3, 3> R;

void setup()
{
  Serial.begin(9600);
  delay(100);
  nvs_access();
  delay(100);

  if (!aht.begin()) {
    Serial.println("Failed to find AHT10/AHT20 chip");
    while (1)
     ;
  }
  aht_temp = aht.getTemperatureSensor();
  aht_humidity = aht.getHumiditySensor();

  actor.attach(servoPin);

  x.zeros();
  x(0) = 500;
  x(1) = 15;
  x(2) = 24;
  x(3) = 0;

  Q.eye(); Q *= 0.01; Q(0, 0) = 1;
  R.eye(); R *= 1; R(0, 0) = 10000; R(1, 1) = 0.01;
  P.eye(); P *= 1000; P(3, 3) = 0.0001;
  I.eye();

  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED)
    delay(500);
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  print("\n\n\n\n");
}

bool irrigation = false;

void observation_update(void){
  /* EKF predict step */
  auto saturated_water_vapor_pressure = 1.455 * x(2) - 3.980;
  auto bet_balance_pressure = (0.853 + 0.03 * log(bet_adsorbed_ratio * x(0) + 1e-5) - 0.0002715 * (bet_adsorbed_ratio * x(0))) * saturated_water_vapor_pressure;
  x(0) += (- dalton_evaporation_constant * (bet_balance_pressure - x(1)) - x(3) * x(0) + (irrigation ? control_rate : 0)) * dt;
  if(x(0) < 0) x(0) = 0;

  Tensor<double, 4, 4> J;
  J.eye();
  J(0, 0) += dt * x(3) + dt * (0.000395 * dalton_evaporation_constant * (x(2) - 2.73474) * (bet_adsorbed_ratio * x(0) - 132.098)) / (x(0) + 1);
  J(0, 1) += dalton_evaporation_constant * dt;
  J(0, 2) += dalton_evaporation_constant * dt * (-1.241 + 0.000395 * bet_adsorbed_ratio * x(0) - 0.052 * log(bet_adsorbed_ratio * x(0) + 1e-5));
  J(0, 3) += x(0) * dt;

  P = matmul(matmul(J, P), transpose(J)) + Q;

  /* Measurement */
  Tensor<double, 3> z, h;

  sensors_event_t temp;
  sensors_event_t humidity;
  aht_temp->getEvent(&temp);
  aht_humidity->getEvent(&humidity);

  z(0) = analogRead(soilHumidityPin);
  z(1) = humidity.relative_humidity / 100;
  z(2) = temp.temperature;

  /* EKF update step */
  h(0) = x(0);
  h(1) = x(1) / (1.455 * x(2) - 3.980);
  h(2) = x(2);

  Tensor<double, 3, 4> H;
  H.zeros();
  H(0, 0) = 1;
  H(1, 1) = 1 / (1.455 * x(2) - 3.980);
  H(1, 2) = - 1.455 * x(1) / ((1.455 * x(2) - 3.980) * (1.455 * x(2) - 3.980));
  H(2, 2) = 1;

  auto y = z - h;
  auto S = matmul(matmul(H, P), transpose(H)) + R;
  auto K = matmul(matmul(P, transpose(H)), inverse(S));

  x = x + matmul(K, y);
  if(x(0) < 0) x(0) = 0;
  if(x(1) < 0) x(1) = 0;
  if(x(3) < 0) x(3) = 0;
  P = matmul(I - matmul(K, H), P);
}

int network_cnt = 0;

void loop(void)
{
  observation_update();

  if(irrigation)
    irrigation = x(0) <= high_thredhold;
  else
    irrigation = x(0) < low_thredhold;

  actor.write(irrigation ? 180 : 0);

  /* network communication  */
  if(network_cnt == 0){
    IPAddress aws_server(18, 144, 64, 220);
    char path[128];
    sprintf(path, "/?hum=%d&pres=%d&temp=%d&infi=%d", (int)(10 * x(0)), (int)(10 * x(1)), (int)(10 * x(2)), (int)(10 * x(3)));
    Serial.println(path);

    WiFiClient wifi_client;
    HttpClient http(wifi_client);

    if(auto err = http.get(aws_server, NULL, 5000, path); err != 0){
      Serial.print("Connect failed: ");
      Serial.println(err);
      http.stop();
      return;
    }

    Serial.println("startedRequest ok");

    if(auto code = http.responseStatusCode(); code < 0){
      Serial.print("Getting response failed: ");
      Serial.println(code);
      http.stop();
      return;
    }

    if(auto err = http.skipResponseHeaders(); err < 0){
        Serial.print("Failed to skip response headers: ");
        Serial.println(err);
        http.stop();
        return;
    }

    char buffer[128];
    int bodyLen = http.contentLength();
    http.read((uint8_t*)buffer, 128);
    buffer[bodyLen] = '\0';
    Serial.println(buffer);

    sscanf(buffer, "%lf, %lf", &low_thredhold, &high_thredhold);
    Serial.print("get new thredhold: ");
    Serial.print(low_thredhold);
    Serial.print(", ");
    Serial.print(high_thredhold);
    Serial.println("\n");

    http.stop();
  }

  Serial.print("soil moisture             : "); Serial.print(x(0)); Serial.print("\t("); Serial.print(P(0, 0)); Serial.print(")\n");
  Serial.print("water vapor pressure (hPa): "); Serial.print(x(1)); Serial.print("\t("); Serial.print(P(1, 1)); Serial.print(")\n");
  Serial.print("air temperature (°C)      : "); Serial.print(x(2)); Serial.print("\t("); Serial.print(P(2, 2)); Serial.print(")\n");
  Serial.print("spermeability             : "); Serial.print(x(3)); Serial.print("\t("); Serial.print(P(3, 3)); Serial.print(")\n");
  Serial.print("irrigation?               : "); Serial.print(irrigation);
  Serial.println("\n\n");

  sleep(3600 * dt);
  network_cnt = (network_cnt + 1) % 10;
}
