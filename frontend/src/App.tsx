import { Routes, Route } from "react-router-dom";
import Home from "@/pages/Home";
import Library from "@/pages/Library";
import ModelLab from "@/pages/ModelLab";

// One <Route> per page in src/pages; BrowserRouter already wraps this in main.tsx.
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/kutuphane" element={<Library />} />
      <Route path="/model" element={<ModelLab />} />
      <Route path="*" element={<Home />} />
    </Routes>
  );
}
