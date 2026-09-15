# 🍕 SliceHub — Pizza Ordering App (MBAX 6418)

A **Gradio** web app where customers can build custom pizzas and place an
order. Customers choose size, crust, sauce, cheeses, toppings, quantity,
sides and drinks — and get a **live price breakdown** plus a confirmation
receipt when they check out.

## Features

- 🍕 **Customize your pizza** — 4 sizes, 5 crusts, 6 sauces, 6 cheeses, 16 toppings
- 📏 **Quantity** — order up to 10 pizzas
- 🧀 **Extras** — garlic seasoning toggle
- 🥗 **Sides & drinks** — 5 sides, 4 drinks
- 🧮 **Live pricing** — subtotal, coupon discount, tip, sales tax, total
- 🎟️ **Coupons** — try code `PIZZA5` for $5 off
- 🧾 **Confirm order** — prints a full itemized receipt

## Run it

### 1. Create the environment (first time only)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

### 2. Launch the app

```bash
.venv/Scripts/python app.py
```

Open the URL printed in the terminal (default `http://127.0.0.1:7860`).

> **Windows note:** use `.venv\Scripts\python` (or `.venv/Scripts/python`)
> to run the interpreter. The venv is already created in this folder.

## The order customizer

| Option      | Choices |
|-------------|---------|
| **Size**    | Small, Medium, Large, X-Large |
| **Crust**   | Hand-Tossed, Thin & Crispy, Pan, Cheese-Stuffed, Gluten-Free |
| **Sauce**   | Tomato, Marinara, White Garlic, Alfredo, Barbecue, Pesto |
| **Cheese**  | Mozzarella, Cheddar, Parmesan, Feta, Blue Cheese, Vegan Cheese |
| **Toppings**| Pepperoni, Sausage, Ham, Bacon, Chicken, Ground Beef, Anchovies, Mushrooms, Onions, Green Peppers, Black Olives, Jalapeños, Tomatoes, Pineapple, Spinach, Basil |
| **Extras**  | Extra garlic seasoning |
| **Sides**   | Garlic Bread, Cheesy Breadsticks, Chicken Wings, Caesar Salad, Mozzarella Sticks |
| **Drinks**  | Soft Drink, Fountain Drink, Bottled Water, Iced Tea |

## Project structure

```
MBAX 6418/
├── app.py            # the Gradio app (menu, pricing, UI, events)
├── requirements.txt  # Python dependencies
└── .venv/            # virtual environment (already set up)
```

## Customizing the menu

All pricing lives at the top of `app.py` in easy-to-edit dictionaries
(`SIZES`, `CRUSTS`, `TOPPINGS`, `SIDES`, `DRINKS`). Change a price, add a
topping, or tweak `TAX_RATE` and the UI updates automatically.
