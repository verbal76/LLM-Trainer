# JNI looks these up by name from native code.
-keep class com.hotatticgames.hag.runtime.NativeBridge { *; }
-keep class com.hotatticgames.hag.runtime.NativeBridge$* { *; }
-keep class com.hotatticgames.hag.runtime.HagException { <init>(...); }
