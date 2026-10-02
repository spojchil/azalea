// Dumps the vanilla data that decides how a right click flows through
// `Minecraft.startUseItem` (block use, item use-on, item use, entity
// interaction), so the client can predict the same outcome without running
// vanilla's block/item/entity logic. The predictor in
// `azalea-client/src/plugins/interact/predict` consumes the generated tables.
//
// For every registered block/item it records which class declares each
// right-click method (overrides only; the base-class default is implied), plus
// the lookup tables those overrides consult (axe stripping, shovel paths,
// waxing, composting, potting, which doors open by hand, which blocks are
// replaceable).
//
// Regenerate (26.1.2 client jar and its libraries, library paths from the
// version json):
//   CP="client.jar:$(find libraries -name '*.jar' | tr '\n' ':')"
//   javac -cp "$CP" -d out RightClickDump.java
//   java -cp "$CP:out" RightClickDump right_click.json
//   python3 entities.py > entities.tsv   (over the decompiled sources)
//   python3 gen.py right_click.json entities.tsv client.jar \
//     ../../azalea-registry/src/builtin.rs > ../../azalea-client/src/plugins/interact/predict/data.rs
import java.io.FileWriter;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Supplier;

import net.minecraft.SharedConstants;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.HangingSignItem;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.SpawnEggItem;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.ComposterBlock;
import net.minecraft.world.level.block.DoorBlock;
import net.minecraft.world.level.block.FlowerPotBlock;
import net.minecraft.world.level.block.TrapDoorBlock;
import net.minecraft.world.level.block.state.properties.BlockSetType;

public class RightClickDump {
    static String declarer(Class<?> start, String name) {
        for (Class<?> c = start; c != null; c = c.getSuperclass()) {
            for (Method m : c.getDeclaredMethods()) {
                if (m.getName().equals(name) && !m.isBridge() && !m.isSynthetic()) {
                    return c.getSimpleName();
                }
            }
        }
        return "-";
    }

    static String block(Block b) {
        return BuiltInRegistries.BLOCK.getKey(b).getPath();
    }

    static String item(Item i) {
        return BuiltInRegistries.ITEM.getKey(i).getPath();
    }

    static Object field(Class<?> owner, String name) throws Exception {
        Field f = owner.getDeclaredField(name);
        f.setAccessible(true);
        Object value = f.get(null);
        return value instanceof Supplier<?> s ? s.get() : value;
    }

    static List<String> blockKeys(Object map) {
        List<String> out = new ArrayList<>();
        for (Object k : ((Map<?, ?>) map).keySet()) {
            out.add(block((Block) k));
        }
        out.sort(null);
        return out;
    }

    static String quote(String s) {
        return "\"" + s + "\"";
    }

    static String list(Collection<String> values) {
        List<String> quoted = new ArrayList<>();
        for (String v : new TreeSet<>(values)) {
            quoted.add(quote(v));
        }
        return "[" + String.join(", ", quoted) + "]";
    }

    public static void main(String[] args) throws Exception {
        SharedConstants.tryDetectVersion();
        Bootstrap.bootStrap();

        Map<String, String> blocks = new TreeMap<>();
        List<String> replaceable = new ArrayList<>();
        List<String> doorsByHand = new ArrayList<>();
        for (Block b : BuiltInRegistries.BLOCK) {
            String itemOn = declarer(b.getClass(), "useItemOn");
            String without = declarer(b.getClass(), "useWithoutItem");
            blocks.put(block(b), "[" + quote(itemOn) + ", " + quote(without) + "]");
            if (b.defaultBlockState().canBeReplaced()) {
                replaceable.add(block(b));
            }
            BlockSetType type = null;
            if (b instanceof DoorBlock door) {
                type = door.type();
            } else if (b instanceof TrapDoorBlock) {
                Field f = TrapDoorBlock.class.getDeclaredField("type");
                f.setAccessible(true);
                type = (BlockSetType) f.get(b);
            }
            if (type != null && type.canOpenByHand()) {
                doorsByHand.add(block(b));
            }
        }

        Map<String, String> items = new TreeMap<>();
        for (Item i : BuiltInRegistries.ITEM) {
            String placed = i instanceof BlockItem bi ? quote(block(bi.getBlock())) : "null";
            items.put(item(i), "{\"use\": " + quote(declarer(i.getClass(), "use"))
                + ", \"use_on\": " + quote(declarer(i.getClass(), "useOn"))
                + ", \"interact_living\": " + quote(declarer(i.getClass(), "interactLivingEntity"))
                + ", \"places\": " + placed
                + ", \"hanging_sign\": " + (i instanceof HangingSignItem)
                + ", \"spawn_egg\": " + (i instanceof SpawnEggItem) + "}");
        }

        List<String> potted = new ArrayList<>();
        for (Object k : ((Map<?, ?>) field(FlowerPotBlock.class, "POTTED_BY_CONTENT")).keySet()) {
            potted.add(block((Block) k));
        }
        List<String> compostables = new ArrayList<>();
        for (Object k : ComposterBlock.COMPOSTABLES.keySet()) {
            compostables.add(item(((net.minecraft.world.level.ItemLike) k).asItem()));
        }

        try (FileWriter out = new FileWriter(args[0])) {
            out.write("{\n");
            out.write("\"blocks\": {");
            out.write(String.join(",\n", blocks.entrySet().stream().map(e -> quote(e.getKey()) + ": " + e.getValue()).toList()));
            out.write("},\n\"items\": {");
            out.write(String.join(",\n", items.entrySet().stream().map(e -> quote(e.getKey()) + ": " + e.getValue()).toList()));
            out.write("},\n");
            out.write("\"strippable\": " + list(blockKeys(field(Class.forName("net.minecraft.world.item.AxeItem"), "STRIPPABLES"))) + ",\n");
            out.write("\"flattenable\": " + list(blockKeys(field(Class.forName("net.minecraft.world.item.ShovelItem"), "FLATTENABLES"))) + ",\n");
            out.write("\"waxable\": " + list(blockKeys(field(Class.forName("net.minecraft.world.item.HoneycombItem"), "WAXABLES"))) + ",\n");
            out.write("\"wax_off\": " + list(blockKeys(field(Class.forName("net.minecraft.world.item.HoneycombItem"), "WAX_OFF_BY_BLOCK"))) + ",\n");
            out.write("\"scrapable\": " + list(blockKeys(field(Class.forName("net.minecraft.world.level.block.WeatheringCopper"), "PREVIOUS_BY_BLOCK"))) + ",\n");
            out.write("\"pottable\": " + list(potted) + ",\n");
            out.write("\"compostable\": " + list(compostables) + ",\n");
            out.write("\"replaceable\": " + list(replaceable) + ",\n");
            out.write("\"opens_by_hand\": " + list(doorsByHand) + "\n");
            out.write("}\n");
        }
    }
}
