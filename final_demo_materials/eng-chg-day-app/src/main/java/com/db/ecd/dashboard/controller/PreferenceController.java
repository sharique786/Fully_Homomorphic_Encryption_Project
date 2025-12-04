package com.db.ecd.dashboard.controller;

import com.db.ecd.dashboard.entity.UserPreference;
import com.db.ecd.dashboard.service.PreferenceService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/preferences")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class PreferenceController {

    private final PreferenceService preferenceService;

    @PostMapping
    public ResponseEntity<List<UserPreference>> submitPreferences(@RequestBody List<UserPreference> preferences) {
        return ResponseEntity.ok(preferenceService.submitPreferences(preferences));
    }

    @GetMapping("/user/{userId}")
    public ResponseEntity<List<UserPreference>> getUserPreferences(@PathVariable Long userId) {
        return ResponseEntity.ok(preferenceService.getUserPreferences(userId));
    }
}
