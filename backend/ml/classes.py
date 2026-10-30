"""Sınıf tanımları — veri seti, eğitim ve çıkarım (inference) aynı listeyi kullanır."""

# Sıra önemlidir: modelin çıkış katmanındaki indeks bu sıraya göre belirlenir.
CLASSES: list[str] = [
    "direnc",
    "led",
    "seramik_kondansator",
    "elektrolitik_kondansator",
    "transistor",
    "diyot",
    "entegre",
    "potansiyometre",
    "bobin",
    "role",
    "sigorta",
    "transformator",
    "buzzer",
    "kristal_osilator",
    "buton",
    "display_7seg",
    "arduino",
    "esp32",
    "stm32",
    "bilinmiyor",
]

# Bileşen olmayan görüntüler (masa, el, klavye...) — canlı kamerada boş kadrajı tanımak için
UNKNOWN = "bilinmiyor"

# Web'den görsel toplamak için arama sorguları (İngilizce sorgular daha çok sonuç verir)
SEARCH_QUERIES: dict[str, list[str]] = {
    "direnc": ["through hole resistor color bands", "carbon film resistor component", "metal film resistor 1/4w", "resistor electronic component photo", "axial resistor close up"],
    "led": ["5mm led diode component", "through hole led red green", "single led light emitting diode two legs", "3mm led electronic component", "led diode close up"],
    "seramik_kondansator": ["ceramic disc capacitor", "ceramic capacitor 104 component", "ceramic capacitor electronic", "monolithic ceramic capacitor", "disc capacitor close up"],
    "elektrolitik_kondansator": ["electrolytic capacitor component", "aluminum electrolytic capacitor", "radial electrolytic capacitor 100uf", "electrolytic capacitor close up", "electrolytic capacitor isolated"],
    "transistor": ["to-92 transistor bc547", "npn transistor 2n2222 component", "to-92 transistor three legs", "bjt transistor electronic component", "transistor close up photo"],
    "diyot": ["1n4007 rectifier diode", "1n4148 diode glass", "rectifier diode component", "zener diode component", "diode electronic component close up"],
    "entegre": ["dip integrated circuit chip", "ne555 ic chip dip8", "dip ic chip electronic component", "integrated circuit dip package", "microcontroller dip chip atmega328"],
    "potansiyometre": ["rotary potentiometer component", "10k potentiometer knob shaft", "trimmer potentiometer", "potentiometer electronic component", "linear potentiometer b10k"],
    "bobin": ["inductor coil component", "toroidal inductor", "radial inductor electronic", "copper coil inductor", "axial inductor component"],
    "role": ["5v relay module component", "electromechanical relay songle", "pcb relay component", "relay electronic component", "automotive relay"],
    "sigorta": ["glass cartridge fuse", "glass fuse 5x20", "electronic fuse component", "blade fuse automotive", "ceramic fuse component"],
    "transformator": ["small power transformer electronic", "step down transformer 220v 12v", "toroidal transformer", "pcb transformer component", "laminated core transformer"],
    "buzzer": ["piezo buzzer component", "active buzzer 5v", "electronic buzzer module", "magnetic buzzer component", "piezo buzzer close up"],
    "kristal_osilator": ["crystal oscillator 16mhz hc-49", "quartz crystal component", "hc-49s crystal", "crystal oscillator electronic component", "crystal resonator 8mhz"],
    "buton": ["tactile push button switch", "tact switch 6x6", "push button switch component", "momentary push button electronic", "tactile switch close up"],
    "display_7seg": ["7 segment display component", "seven segment led display", "single digit 7 segment display", "4 digit 7 segment display", "7 segment display red"],
    "arduino": ["arduino uno board", "arduino uno r3 isolated", "arduino nano board", "arduino mega 2560 board", "arduino leonardo board", "arduino nano close up", "arduino uno top view", "arduino uno smd", "arduino nano every", "arduino pro mini board"],
    "esp32": ["esp32 devkit board", "esp32 wroom development board", "esp32 devkit v1", "esp32 dev module isolated", "esp32 board top view", "esp32 30 pin board", "esp32 38 pin development board", "esp32 s3 devkit", "esp32 board close up", "esp32 c3 mini board"],
    "stm32": ["stm32 blue pill board", "stm32f103c8t6 board", "stm32 nucleo board", "stm32 black pill board", "stm32 development board", "stm32f4 discovery board", "stm32 nucleo-64", "stm32f401 black pill", "stm32 blue pill close up", "stm32 nucleo f411re"],
}

# CLIP ile otomatik temizlik için sınıf metinleri
CLIP_PROMPTS: dict[str, str] = {
    "direnc": "a photo of an axial resistor with color bands",
    "led": "a photo of a light emitting diode LED with two legs",
    "seramik_kondansator": "a photo of a ceramic disc capacitor",
    "elektrolitik_kondansator": "a photo of a cylindrical electrolytic capacitor",
    "transistor": "a photo of a transistor with three legs",
    "diyot": "a photo of a small cylindrical diode with a stripe",
    "entegre": "a photo of an integrated circuit chip",
    "potansiyometre": "a photo of a potentiometer with a knob shaft",
    "bobin": "a photo of an inductor coil",
    "role": "a photo of an electromechanical relay",
    "sigorta": "a photo of a fuse or a miniature circuit breaker",
    "transformator": "a photo of an electrical transformer",
    "buzzer": "a photo of a round black piezo buzzer",
    "kristal_osilator": "a photo of a metal quartz crystal oscillator",
    "buton": "a photo of a tactile push button switch",
    "display_7seg": "a photo of a seven segment LED display",
    "arduino": "a photo of an Arduino microcontroller board",
    "esp32": "a photo of an ESP32 wifi development board",
    "stm32": "a photo of an STM32 microcontroller development board",
}

# Gürültü sınıfları: bunlara en çok benzeyen görseller atılır
CLIP_NEGATIVE_PROMPTS: list[str] = [
    "a photo of a printed circuit board full of many components",
    "a schematic circuit diagram drawing",
    "a photo of a large machine",
    "a photo of text or a logo",
    "a photo of a person",
    "a photo of a light strip or a lamp",
    "a photo of a building or a room",
]

# Az görseli olan sınıflar için ek sorgular (python -m ml.build_dataset --more)
EXTRA_QUERIES: dict[str, list[str]] = {
    "direnc": ["220 ohm resistor", "10k resistor", "1k ohm resistor close up", "resistor isolated white background", "blue metal film resistor", "beige carbon resistor bands", "resistors pack", "resistor on breadboard", "4 band resistor", "5 band resistor"],
    "seramik_kondansator": ["104 ceramic capacitor", "ceramic capacitor 22pf", "blue ceramic capacitor", "orange disc capacitor", "ceramic capacitor isolated white background", "high voltage ceramic disc capacitor", "ceramic capacitor on breadboard", "capacitor 103 ceramic"],
    "buton": ["tactile button 12x12", "push button 4 pin", "tact switch on breadboard", "arduino push button", "micro push button switch", "tactile switch isolated white background", "6x6x5 tact switch", "push button colored caps"],
    "buzzer": ["active buzzer arduino", "passive buzzer", "piezo buzzer 12mm", "buzzer on breadboard", "buzzer isolated white background", "5v buzzer black", "piezo buzzer pcb", "dc buzzer 12v"],
    "kristal_osilator": ["16mhz crystal arduino", "8mhz crystal oscillator", "quartz crystal hc49", "crystal oscillator isolated", "12mhz crystal", "crystal oscillator on pcb", "4mhz crystal", "20mhz crystal"],
    "display_7seg": ["7 segment display arduino", "common cathode 7 segment", "common anode 7 segment display", "7 segment display isolated", "green 7 segment display", "7 segment display on breadboard", "0.56 inch 7 segment", "blue 7 segment display"],
    "bobin": ["toroid inductor copper", "100uh inductor", "radial power inductor", "choke coil", "ferrite core inductor", "smd power inductor"],
    "sigorta": ["5x20mm glass fuse", "fast blow glass fuse", "blade fuse set", "ceramic tube fuse", "fuse isolated white background"],
    "diyot": ["1n4007 diode close up", "1n5819 schottky diode", "diodes isolated white background", "zener diode glass", "rectifier diode axial black"],
    "transistor": ["bc547 transistor", "2n3904 transistor", "tip120 transistor", "irf540 mosfet", "transistor isolated white background"],
    "potansiyometre": ["b10k potentiometer", "potentiometer with knob", "trimpot blue", "rotary potentiometer isolated"],
    "elektrolitik_kondansator": ["100uf capacitor", "1000uf capacitor", "electrolytic capacitors isolated white background", "capacitor 470uf 25v"],
    "led": ["5mm red led", "led diode isolated white background", "rgb led 4 pin", "led green 5mm"],
    "entegre": ["ne555 chip", "lm358 dip", "74hc595 ic", "atmega328p dip"],
    "role": ["songle relay 5v", "relay module arduino", "12v relay", "relay isolated white background"],
    "transformator": ["12v transformer", "small transformer isolated", "toroidal transformer isolated", "mains transformer"],
}
