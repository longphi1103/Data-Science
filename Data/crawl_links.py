import asyncio
import random
import os
import pandas as pd
from playwright.async_api import async_playwright

# ==========================================
# CẤU HÌNH GIAI ĐOẠN 2: CHẾ ĐỘ 20/20
# ==========================================
INPUT_FILE = "danh_sach_link.txt"     
START_INDEX = 20
END_INDEX = 30

# Lấy chính xác 20 cái mỗi loại (nếu quán có đủ)
MAX_PER_TYPE = 20                     
MIN_LENGTH = 30                       
MAX_LENGTH = 1000                     

SUMMARY_FILE = "thong_ke_nha_hang.csv"   
REVIEWS_FILE = "du_lieu_reviews.txt"     

async def human_delay(a=1.2, b=2.5):
    await asyncio.sleep(random.uniform(a, b))

async def change_sort(page, sort_type="Lowest"):
    """Đổi bộ lọc: 'Lowest' (Thấp nhất) hoặc 'Highest' (Cao nhất)"""
    try:
        sort_btn = await page.wait_for_selector(
            "button:has-text('Phù hợp nhất'), button:has-text('Mới nhất'), button:has-text('Xếp hạng thấp nhất'), button:has-text('Xếp hạng cao nhất'), button[aria-label*='Sắp xếp']", 
            timeout=5000
        )
        await sort_btn.scroll_into_view_if_needed()
        await sort_btn.click()
        await human_delay(1.5, 2.5) 
        
        if sort_type == "Lowest":
            option_locator = page.locator('div[role="menuitemradio"]:has-text("Xếp hạng thấp nhất")')
        else:
            option_locator = page.locator('div[role="menuitemradio"]:has-text("Xếp hạng cao nhất")')
            
        await option_locator.wait_for(state="visible", timeout=5000)
        await option_locator.evaluate("node => node.click()")
        await human_delay(3, 5) 
        return True
    except: return False

async def scrape_current_view(page, target_count, current_reviews_set):
    """Cào review, ưu tiên bấm 'Xem bản dịch' để lấy tiếng Việt"""
    count_before = len(current_reviews_set)
    last_count = 0
    same_count = 0
    
    try:
        await page.wait_for_selector("div.jftiEf", timeout=7000)
    except: return 0

    while (len(current_reviews_set) - count_before) < target_count:
        blocks = await page.query_selector_all("div.jftiEf")
        for b in blocks:
            try:
                # Cô lập vùng review của khách
                review_container = await b.query_selector("div.MyEned")
                if not review_container: continue

                # 1. Bấm 'Xem thêm' nếu review quá dài
                more_btn = await review_container.query_selector("button:has-text('Xem thêm'), button:has-text('More')")
                if more_btn:
                    await more_btn.click()
                    await human_delay(0.4, 0.7)

                # 🌟 MỚI: Kiểm tra và bấm 'Xem bản dịch' (View Translation)
                # Google thường dùng text 'Xem bản dịch' hoặc 'See translation'
                translate_btn = await review_container.query_selector("button:has-text('Xem bản dịch'), button:has-text('See translation')")
                if translate_btn:
                    await translate_btn.click()
                    # Phải đợi một chút để Google đổi text sang tiếng Việt
                    await human_delay(0.8, 1.2)

                # 2. Lấy Text (Lúc này đã là tiếng Việt sau khi dịch)
                text_el = await review_container.query_selector("span.wiI7pd")
                if text_el:
                    review_text = await text_el.inner_text()
                    if MIN_LENGTH <= len(review_text) <= MAX_LENGTH:
                        # Ép về 1 dòng duy nhất
                        clean_text = review_text.replace('\n', ' ').strip()
                        current_reviews_set.add(clean_text)
                        
                        if (len(current_reviews_set) - count_before) >= target_count: break
            except: continue

        if (len(current_reviews_set) - count_before) >= target_count: break

        await page.mouse.wheel(0, 1500)
        await human_delay(2, 3)

        current_on_screen = len(await page.query_selector_all("div.jftiEf"))
        if current_on_screen == last_count:
            same_count += 1
            if same_count > 3: break
        else: same_count = 0
        last_count = current_on_screen
    
    return len(current_reviews_set) - count_before

async def run():
    if not os.path.exists(INPUT_FILE): return
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        urls_list = [line.strip() for line in f if line.strip()]
    ca_hien_tai = urls_list[START_INDEX:END_INDEX]
    
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=f"user_data_reviews_{START_INDEX}",
            headless=False,
            args=["--start-maximized", "--disable-blink-features=AutomationControlled"],
            locale="vi-VN"
        )
        page = await browser.new_page()

        for idx, url in enumerate(ca_hien_tai):
            print(f"\n[{idx+1}/{len(ca_hien_tai)}] Quán số {START_INDEX + idx}")
            try:
                await page.goto(url)
                await human_delay(4, 6)
                
                name_el = await page.wait_for_selector("h1", timeout=5000)
                restaurant_name = await name_el.inner_text()
                
                btn = await page.wait_for_selector("//button[contains(., 'Bài đánh giá') or contains(., 'Reviews')]", timeout=5000)
                await btn.click()
                await human_delay(2, 4)
                
                final_reviews = set()

                # LƯỢT 1: Lấy 20 Xấu
                print("⏬ Đang lấy 20 đánh giá Xấu...")
                bad_count = 0
                if await change_sort(page, "Lowest"):
                    bad_count = await scrape_current_view(page, MAX_PER_TYPE, final_reviews)
                
                # LƯỢT 2: Lấy 20 Tốt
                print("⏫ Đang lấy 20 đánh giá Tốt...")
                good_count = 0
                if await change_sort(page, "Highest"):
                    good_count = await scrape_current_view(page, MAX_PER_TYPE, final_reviews)

                print(f"📊 Kết quả: {bad_count} Xấu, {good_count} Tốt. Tổng: {len(final_reviews)}")

                if final_reviews:
                    # 1. Thống kê (CSV)
                    summary_df = pd.DataFrame([{"restaurant_name": restaurant_name, "bad": bad_count, "good": good_count, "total": len(final_reviews)}])
                    summary_df.to_csv(SUMMARY_FILE, mode='a', index=False, header=not os.path.isfile(SUMMARY_FILE), encoding="utf-8-sig")

                    # 2. Lưu Review (TXT - 1 dòng/review)
                    with open(REVIEWS_FILE, mode='a', encoding='utf-8') as f:
                        for r in final_reviews:
                            f.write(r + '\n')
                    print(f"💾 Đã lưu xong.")

            except Exception as e:
                print(f"❌ Lỗi tại quán này: {e}")
                continue
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())