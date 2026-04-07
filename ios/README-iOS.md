# AXIOM iOS App

Full SwiftUI app that connects to the AXIOM FastAPI backend.

## Requirements
- Xcode 15+
- iOS 16+ (required for Swift Charts)
- macOS Ventura or later

## Setup

### 1. Create Xcode Project
```
File → New → Project → App
Product Name: AXIOM
Bundle ID: com.yourname.axiom
Interface: SwiftUI
Language: Swift
Minimum Deployment: iOS 16.0
```

### 2. Add files
Copy the entire `ios/AXIOM/` folder into your Xcode project.

### 3. Configure API URL
Edit `Services/APIService.swift`:
```swift
static let baseURL = "https://YOUR-RAILWAY-APP.railway.app"
```

### 4. Enable Push Notifications
In Xcode → Signing & Capabilities → + Capability → Push Notifications

### 5. Add to AppDelegate
```swift
// AppDelegate.swift
import UIKit
class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        Task { await NotificationService.shared.sendTokenToBackend(deviceToken) }
    }
}
```

## Architecture
```
AXIOM iOS
├── AXIOMApp.swift              ← Entry point + AppStore
├── Views/
│   └── DashboardView.swift     ← All views (Dashboard, Predictions, News, Backtest)
├── Models/
│   └── DataModels.swift        ← Codable structs matching API
├── Services/
│   ├── APIService.swift        ← All API calls (async/await)
│   └── NotificationService.swift ← APNs + local alerts
└── Components/                 ← Reusable UI components
```

## On-Device ML (Advanced — Phase 2)
Convert trained models to Core ML format:

```python
# On your Mac, in the repo root:
import coremltools as ct
import pickle, xgboost as xgb

# Load XGBoost model
with open("trading_system/models/PLTR_v2_xgb.pkl","rb") as f:
    model = pickle.load(f)

# Convert
cml = ct.converters.xgboost.convert(model, feature_names=FEAT_COLS,
                                     target_feature_name="signal")
cml.save("ios/AXIOM/Models/PLTR_XGB.mlmodel")
```

Then in Swift:
```swift
import CoreML
let model = try PLTR_XGB(configuration: MLModelConfiguration())
let prediction = try model.prediction(from: featureProvider)
```

## Deployment to App Store
1. Archive → Xcode → Product → Archive
2. Distribute → TestFlight first
3. App Store Connect → Submit for review
4. Review time: ~24-48 hours

## Features
- [x] Real-time signal cards (PLTR, AAPL, NVDA, TSLA)
- [x] LSTM 4-week price forecast chart (Swift Charts)
- [x] Intraday direction prediction
- [x] Bull/Base/Bear scenarios
- [x] Live news feed with sentiment
- [x] Backtest performance stats
- [x] Daily pre-market push notifications (6:30am)
- [x] Signal-change push alerts
- [ ] Phase 2: On-device Core ML inference
- [ ] Phase 2: WidgetKit lock screen widget
- [ ] Phase 2: Apple Watch complication
