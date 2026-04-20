#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_netif.h"

static const char *TAG = "RX_ESP32";
#define WIFI_CHANNEL 6

static uint32_t csi_count = 0;

// CSI callback
static void csi_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info->buf || info->len == 0) return;
    
    csi_count++;
    
    printf("\n[CSI #%lu] Len:%d RSSI:%d\n", csi_count, info->len, info->rx_ctrl.rssi);
    
    printf("Amp: ");
    int max = (info->len / 2 < 64) ? info->len / 2 : 64;
    for (int i = 0; i < max; i++) {
        int8_t real = info->buf[i*2];
        int8_t imag = info->buf[i*2+1];
        float amp = sqrtf(real*real + imag*imag);
        printf("%.1f ", amp);
        if ((i+1) % 16 == 0 && i < max-1) printf("\n     ");
    }
    printf("\n");
}

void app_main(void) {
    ESP_LOGI(TAG, "CSI Receiver Started");
    
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_cb, NULL));
    
    wifi_csi_config_t csi_config = {
        .lltf_en = 1,
        .htltf_en = 1,
        .manu_scale = 0,
        .shift = 1,
    };
    esp_wifi_set_csi_config(&csi_config);
    
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_csi(true);
    
    esp_wifi_start();
    
    ESP_LOGI(TAG, "CSI Ready on channel %d", WIFI_CHANNEL);
}