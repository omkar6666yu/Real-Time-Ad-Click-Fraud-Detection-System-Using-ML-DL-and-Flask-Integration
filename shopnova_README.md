# ShopNova v2 — Fraud Feature Demo Website

A complete fake e-commerce website with:
- Draggable BOT ball that fires bot clicks on any element
- Live fraud signal panel showing all 18 features firing in real time
- 20 trigger buttons — one per fraud feature + legit + clear
- Direct link to Flask /realtime monitor page

## How to run

```
1. python app.py          (start Flask)
2. Open shopnova.html     (double-click, opens in browser)
3. Set API URL to:        http://127.0.0.1:5000/api/predict
```

## Demo for viva

Open side by side:
- LEFT window:  shopnova.html
- RIGHT window: http://127.0.0.1:5000/realtime

Click "F01 Click-burst" — watch fraud counter spike on both screens.
Click "F03 Device mismatch" — see "Desktop+Android OS mismatch" badge.
Click "F07 Night activity" — see "Hour: 2am, Off-hours flag=1" badge.
Click "Human click (legit)" — green LEGITIMATE result for contrast.

Drag the BOT ball onto "Add to cart" — it auto-fires F01 click-burst continuously.
