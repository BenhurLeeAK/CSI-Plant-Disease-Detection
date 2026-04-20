#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_mac.h"
#include "esp_netif.h"

static const char *TAG = "TX_ESP32";

#define WIFI_CHANNEL 6
#define TX_PACKET_RATE_MS 10  // 100 packets per second

// Simple 802.11 packet template (QoS Data)
// Reduced to correct size - 32 bytes total
uint8_t packet[32] = {
    0x08, 0x02, 0x00, 0x00,  // Frame control: QoS Data
    0x00, 0x00,              // Duration
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,  // Destination MAC (broadcast)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // Source MAC (will be filled)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // BSSID (will be filled)
    0x00, 0x00,              // Sequence control
    0x00, 0x00,              // QoS control
    // Simple payload
    0x50, 0x4C, 0x54, 0x44,  // "PLTD" - Plant Disease Test
    0x00, 0x01, 0x02, 0x03   // Counter placeholder
};

void wifi_init_tx(void) {
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    // Initialize network interface
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    
    // Initialize WiFi
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    
    // Set channel
    ESP_ERROR_CHECK(esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE));
    
    // Get MAC address and fill packet
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    memcpy(&packet[10], mac, 6);  // Source MAC
    memcpy(&packet[16], mac, 6);  // BSSID
    
    ESP_LOGI(TAG, "WiFi initialized on channel %d", WIFI_CHANNEL);
    ESP_LOGI(TAG, "MAC Address: %02X:%02X:%02X:%02X:%02X:%02X", 
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

void transmitter_task(void *pvParameters) {
    TickType_t last_wake_time = xTaskGetTickCount();
    uint32_t packet_count = 0;
    
    while(1) {
        // Send packet
        esp_err_t ret = esp_wifi_80211_tx(WIFI_IF_STA, packet, sizeof(packet), false);
        if (ret == ESP_OK) {
            packet_count++;
            if (packet_count % 1000 == 0) {
                // Fix: Use %lu for uint32_t
                ESP_LOGI(TAG, "Sent %lu packets", (unsigned long)packet_count);
            }
        } else if (ret != ESP_ERR_WIFI_NOT_STARTED) {
            ESP_LOGE(TAG, "Failed to send packet: %s", esp_err_to_name(ret));
        }
        
        // Maintain consistent packet rate
        vTaskDelayUntil(&last_wake_time, TX_PACKET_RATE_MS / portTICK_PERIOD_MS);
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "Starting Plant Disease Detection - Transmitter");
    
    wifi_init_tx();
    
    // Start transmitter task
    xTaskCreate(transmitter_task, "tx_task", 4096, NULL, 5, NULL);
    
    ESP_LOGI(TAG, "Transmitter running. Sending packets every %d ms", TX_PACKET_RATE_MS);
}