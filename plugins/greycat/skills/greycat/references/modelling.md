# GreyCat Modelling

## Abstract Types & Inheritance

Abstract types enable polymorphism with both concrete and abstract methods. Only abstract types can be extended.

```gcl
abstract type Building {
    address: String;
    year_built: int;

    fn calculate_tax(): float;  // Abstract - must be implemented
    fn get_age(): int { return 2024 - year_built; }  // Concrete - shared
}

type House extends Building {
    bedrooms: int;
    fn calculate_tax(): float { return bedrooms * 100.0; }
}

type Commercial extends Building {
    square_meters: float;
    fn calculate_tax(): float { return square_meters * 5.0; }
}
```

### Polymorphic Storage

```gcl
var buildings_by_address: nodeIndex<String, node<Building>>;

// Store different subtypes in same index
var house = node<House>{ House { address: "123 Main St", year_built: 2000, bedrooms: 3 }};
buildings_by_address.set(house->address, house);

var shop = node<Commercial>{ Commercial { address: "456 Market St", year_built: 2010, square_meters: 150.0 }};
buildings_by_address.set(shop->address, shop);

// Polymorphic iteration
for (address, building in buildings_by_address) {
    var tax = building->calculate_tax();  // Calls correct implementation
}

// Type-narrowing
var building = buildings_by_address.get("123 Main St").resolve();
if (building is House) {}
```

**Key rules**: abstract methods must be implemented, concrete methods cannot be overridden, `is` for type check, `as` for cast

