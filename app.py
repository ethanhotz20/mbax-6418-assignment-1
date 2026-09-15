"""
MBAX 6418 — Pizza Ordering App
A Gradio web app where customers can build and order custom pizzas
(choose sizes, crusts, sauces, cheeses, toppings, and add sides & drinks).

Run with:
    .venv/Scripts/python app.py
Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import gradio as gr

# --------------------------------------------------------------------------- #
# Menu / pricing data
# --------------------------------------------------------------------------- #
# Sizes -> base price + name
SIZES = {
    "Small (10\")":   {"price": 8.99},
    "Medium (12\")":  {"price": 10.99},
    "Large (14\")":   {"price": 12.99},
    "X-Large (16\")": {"price": 14.99},
}

# Crusts -> extra cost (handmade = free, stuffed costs more)
CRUSTS = {
    "Hand-Tossed":   0.00,
    "Thin & Crispy": 0.00,
    "Pan":           1.00,
    "Cheese-Stuffed": 2.50,
    "Gluten-Free":   3.00,
}

# Sauces -> free
SAUCES = [
    "Tomato", "Marinara", "White Garlic", "Alfredo", "Barbecue", "Pesto",
]

# Cheeses -> free (stack them if you like)
CHEESES = [
    "Mozzarella", "Cheddar", "Parmesan", "Feta", "Blue Cheese", "Vegan Cheese",
]

# Toppings -> price per topping applied once
TOPPINGS = [
    # Meats
    "Pepperoni", "Sausage", "Ham", "Bacon", "Chicken", "Ground Beef", "Anchovies",
    # Veggies
    "Mushrooms", "Onions", "Green Peppers", "Black Olives", "Jalapeños",
    "Tomatoes", "Pineapple", "Spinach", "Basil",
]

SIDES = {
    "Garlic Bread":       4.99,
    "Cheesy Breadsticks": 5.99,
    "Chicken Wings (8)": 8.99,
    "Caesar Salad":       6.49,
    "Mozzarella Sticks": 6.99,
}

DRINKS = {
    "Soft Drink (20oz)": 2.49,
    "Fountain Drink":    2.99,
    "Bottled Water":     1.99,
    "Iced Tea":          2.49,
    "No Drink":          0.00,
}

TAX_RATE = 0.0825  # 8.25% sales tax

# --------------------------------------------------------------------------- #
# Pricing logic
# --------------------------------------------------------------------------- #
def build_order(
    size, crust, sauce, cheeses, toppings,
    quantity, garlic_extra, sides, drinks, tips_pct, coupon_code
):
    """Compute the full itemized order from the widget selections."""

    # 1. Pizza base price
    size_price = SIZES[size]["price"]
    crust_extra = CRUSTS[crust]

    # 2. Toppings: a size-based bump per topping on top of the base cost
    #    (charged once each, regardless of how many pizzas)
    per_topping = 1.25
    topping_total = len(toppings) * per_topping

    pizza_subtotal = (size_price + crust_extra + topping_total) * quantity

    # 3. Extras
    extra_total = 0.0
    if garlic_extra:
        extra_total += 1.50

    # 4. Sides
    side_total = sum(SIDES[s] for s in sides)

    # 5. Drinks
    drink_total = DRINKS[drinks]

    item_subtotal = round(pizza_subtotal + extra_total + side_total + drink_total, 2)

    # 6. Coupon (simple flat $ off, e.g. "PIZZA5")
    discount = 0.0
    coupon_msg = ""
    code = (coupon_code or "").strip().upper()
    if code == "PIZZA5":
        discount = round(min(5.00, item_subtotal), 2)
        coupon_msg = f"Coupon {code} applied: -${discount:.2f}"
    elif code:
        coupon_msg = f"Coupon '{code}' is not valid."

    after_coupon = round(item_subtotal - discount, 2)
    tip = round(after_coupon * tips_pct, 2)
    tax = round(after_coupon * TAX_RATE, 2)
    total = round(after_coupon + tip + tax, 2)

    # 7. Build a readable order summary
    lines = []
    lines.append(f"🍕 {quantity} x {size} Pizza")
    lines.append(f"    Crust: {crust} ({crust_extra:+.2f})")
    lines.append(f"    Sauce: {sauce}")
    lines.append(f"    Cheese: {', '.join(cheeses) if cheeses else 'None'}")
    lines.append(
        f"    Toppings ({len(toppings)}): {', '.join(toppings) if toppings else 'None'}"
    )
    if garlic_extra:
        lines.append("    Extra: + Garlic Seasoning (+$1.50)")
    if sides:
        lines.append(f"    Sides: {', '.join(sides)}")
    if drinks != "No Drink":
        lines.append(f"    Drink: {drinks}")

    order = "\n".join(lines)

    # 8. Price breakdown
    breaks = []
    breaks.append(f"Pizza subtotal (base + crust + toppings) x{quantity} .. ${pizza_subtotal:.2f}")
    if extra_total:
        breaks.append(f"Extra garlic seasoning ................. ${extra_total:.2f}")
    if side_total:
        breaks.append(f"Sides ................................... ${side_total:.2f}")
    if drink_total:
        breaks.append(f"Drinks .................................. ${drink_total:.2f}")
    breaks.append(f"Item subtotal ........................... ${item_subtotal:.2f}")
    if discount:
        breaks.append(f"Coupon discount ......................... -${discount:.2f}")
    if tip:
        breaks.append(f"Tip ({int(tips_pct*100)}%) .................................... ${tip:.2f}")
    breaks.append(f"Sales tax ({TAX_RATE*100:.2f}%) .............................. ${tax:.2f}")
    breaks.append(f"{'='*54}")
    breaks.append(f"TOTAL .................................... ${total:.2f}")

    summary = "\n".join(breaks)
    checkout = (
        f"Thank you! 🍕 Your order is confirmed.\n\n"
        f"{order}\n\n"
        f"{summary}\n\n"
        f"{coupon_msg}"
    )

    return order, summary, f"${total:.2f}", checkout


# --------------------------------------------------------------------------- #
# UI builders
# --------------------------------------------------------------------------- #
def build_app():
    with gr.Blocks(title="SliceHub 🍕 — Pizza Ordering") as demo:
        gr.Markdown(
            "# 🍕 SliceHub — Build Your Pizza\n"
            "Customize your pie exactly how you like it, add sides & drinks, "
            "then see a live breakdown and check out."
        )

        with gr.Row():
            # ------------------------- LEFT: pizza ------------------------- #
            with gr.Column(scale=3):
                gr.Markdown("### 1. Choose Your Pizza")
                size = gr.Radio(
                    list(SIZES.keys()), value=list(SIZES.keys())[1],
                    label="Size", info="Baked fresh to order."
                )
                crust = gr.Radio(
                    list(CRUSTS.keys()), value=list(CRUSTS.keys())[0],
                    label="Crust",
                )
                sauce = gr.Dropdown(
                    SAUCES, value=SAUCES[0],
                    label="Sauce",
                )
                cheeses = gr.CheckboxGroup(
                    CHEESES, value=["Mozzarella"],
                    label="Cheeses", info="Pick any combination.",
                )
                toppings = gr.CheckboxGroup(
                    TOPPINGS, label="Toppings",
                )
                quantity = gr.Slider(
                    1, 10, value=1, step=1,
                    label="Number of pizzas",
                )
                garlic_extra = gr.Checkbox(
                    label="Extra garlic seasoning (+$1.50)", value=False,
                )

            # ------------------------ RIGHT: checkout ---------------------- #
            with gr.Column(scale=2):
                gr.Markdown("### 2. Sides & Drinks")
                sides = gr.CheckboxGroup(
                    list(SIDES.keys()), label="Sides",
                )
                drinks = gr.Radio(
                    list(DRINKS.keys()), value="No Drink", label="Drinks",
                )

                gr.Markdown("### 3. Checkout")
                tips_pct = gr.Slider(
                    0, 0.25, value=0.15, step=0.05,
                    label="Tip", info="15% recommended 😉",
                )
                coupon = gr.Textbox(label="Coupon code", placeholder="Try PIZZA5")

                order_summary = gr.Textbox(
                    label="Your Order", interactive=False, lines=6,
                )
                price_break = gr.Textbox(
                    label="Price Breakdown", interactive=False, lines=12,
                )
                total_out = gr.Textbox(
                    label="Total", interactive=False,
                    elem_classes="total", show_label=True,
                )
                checkout_btn = gr.Button("Confirm Order 🍕", variant="primary")
                receipt = gr.Textbox(
                    label="Receipt", interactive=False, lines=12, visible=True,
                )

        # --------------------------- live updates -------------------------- #
        all_inputs = [
            size, crust, sauce, cheeses, toppings, quantity,
            garlic_extra, sides, drinks, tips_pct, coupon,
        ]

        # live-update the order summary, breakdown, and total as the
        # customer changes anything
        def update_order(*a):
            order, brk, total, _ = build_order(*a)
            return order, brk, total

        for inp in all_inputs:
            inp.change(update_order, all_inputs, [order_summary, price_break, total_out])

        checkout_btn.click(
            lambda *a: build_order(*a)[3],
            all_inputs, receipt,
        )

        # Initial render
        demo.load(
            lambda *a: list(build_order(*a)[:3]),
            all_inputs, [order_summary, price_break, total_out],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch(inbrowser=True, theme=gr.themes.Soft())
