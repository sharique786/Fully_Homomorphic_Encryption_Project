package com.db.ecd.dashboard;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;

@SpringBootApplication
@EnableCaching
public class ECDDashboardApplication {
    public static void main(String[] args) {
        SpringApplication.run(ECDDashboardApplication.class, args);
    }
}
