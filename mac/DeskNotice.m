#import <AppKit/AppKit.h>
#import <UserNotifications/UserNotifications.h>

@interface NoticeDelegate : NSObject <UNUserNotificationCenterDelegate>
@end
@implementation NoticeDelegate
- (void)userNotificationCenter:(UNUserNotificationCenter *)center willPresentNotification:(UNNotification *)notification withCompletionHandler:(void (^)(UNNotificationPresentationOptions))handler {
    handler(UNNotificationPresentationOptionBanner | UNNotificationPresentationOptionList);
}
@end
static NoticeDelegate *notificationDelegate;

static void finish(NSString *message, int code) {
    puts(message.UTF8String); fflush(stdout); exit(code);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyAccessory];
        NSArray<NSString *> *args = NSProcessInfo.processInfo.arguments;
        UNUserNotificationCenter *center = UNUserNotificationCenter.currentNotificationCenter;
        if (args.count > 1 && [args[1] isEqualToString:@"--delivered"]) {
            [center getDeliveredNotificationsWithCompletionHandler:^(NSArray<UNNotification *> *notifications) {
                NSMutableArray *identifiers = [NSMutableArray new];
                for (UNNotification *notification in notifications) [identifiers addObject:notification.request.identifier];
                NSData *json = [NSJSONSerialization dataWithJSONObject:identifiers options:0 error:nil];
                finish([[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding], 0);
            }];
            [app run]; return 0;
        }
        notificationDelegate = [NoticeDelegate new];
        center.delegate = notificationDelegate;
        void (^deliver)(void) = ^{
            UNMutableNotificationContent *content = [UNMutableNotificationContent new];
            content.title = args.count > 2 ? args[2] : @"桌间 · 本地提醒";
            content.body = args.count > 3 ? args[3] : @"Mac 系统通知已连接。久坐与定时提醒会显示在这里。";
            content.sound = nil;
            NSString *identifier = args.count > 4 ? args[4] : @"deskside-connection-test";
            UNNotificationRequest *request = [UNNotificationRequest requestWithIdentifier:identifier content:content trigger:nil];
            [center addNotificationRequest:request withCompletionHandler:^(NSError *error) {
                if (error) finish(error.localizedDescription, 1);
                dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{ finish(@"submitted", 0); });
            }];
        };
        [center getNotificationSettingsWithCompletionHandler:^(UNNotificationSettings *settings) {
            if (args.count > 1 && [args[1] isEqualToString:@"--status"]) {
                finish([NSString stringWithFormat:@"{\"authorization\":%ld,\"alerts\":%ld,\"style\":%ld,\"notification_center\":%ld}", (long)settings.authorizationStatus, (long)settings.alertSetting, (long)settings.alertStyle, (long)settings.notificationCenterSetting], 0);
            }
            if (settings.authorizationStatus == UNAuthorizationStatusNotDetermined) {
                [center requestAuthorizationWithOptions:UNAuthorizationOptionAlert completionHandler:^(BOOL allowed, NSError *error) {
                    if (allowed) deliver(); else finish(@"notification_permission_denied", 2);
                }];
            } else if (settings.authorizationStatus == UNAuthorizationStatusDenied) {
                finish(@"notification_permission_denied", 2);
            } else { deliver(); }
        }];
        [app run];
    }
    return 0;
}
